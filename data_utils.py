import re
import string
import os
import pandas as pd
import torch
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer
from functools import partial
from plot_utils import plot_column_histogram
from train_utils import get_sampler

CACHED_STOP_WORDS = stopwords.words("english")
SEED = 42069666


class TextDataset(Dataset):
    """
    This class is used to create a dataset from a dataframe.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = 128,
            label_col: str = 'label',
            data_col: str = 'text',
            device: torch.device = torch.device('cpu')
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.device = device

        # Column names for the data_utils and the labels
        self.data_col = data_col
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row[self.data_col]
        text = text[-self.max_token_len:]  # Use the last max_token_len tokens.
        label = data_row[self.label_col]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            text=text,
            label=label,
            text_tokenized=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten()
        )


def get_dataloader(tokenizer, df, config, device=torch.device('cpu'), sampler=None):
    """Create dataloader for the given dataframe."""
    ds = TextDataset(df, tokenizer=tokenizer, max_token_len=config.model.max_seq_length, device=device)
    # TODO Pass somehow seed to the dataloader.
    dr = DataLoader(ds, batch_size=config.train.batch_size,
                    shuffle=True if sampler is None else False, sampler=sampler)
    return dr


def clean_text(text):
    """ Preprocess the text.
        Includes:
            - Convert text to lowercase
            - Remove newlines
            - Remove numbers
            - Remove punctuation
            - Remove stopwords
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in CACHED_STOP_WORDS])
    return text


def create_segments(text, segment_length=256, overlap=50):
    """
    Splits the input text into overlapping segments of a specified length.
    """
    words = text.split()
    segments = []
    start = 0
    while start < len(words):
        end = start + segment_length
        segments.append(' '.join(words[start:end]))
        start += (segment_length - overlap)
    return segments


def preprocess_dataframe(df, config):
    """
    Preprocess the dataframe. This includes:
        - label mapping to 0 and 1 for non-news and news respectively.
        - Change the column names to be more descriptive. And drop the original columns.
        - Preprocess the text (lowercase, remove punctuation, remove stopwords).
    """
    class_column = config.data.class_column
    positive_class = config.data.data_class[1]
    negative_class = config.data.data_class[0]
    label_map = {positive_class: 1, negative_class: 0}
    # Binaries the content type column. i.e 1 for news and 0 for non-news.
    df[class_column] = df[class_column].apply(
        lambda x: positive_class if positive_class in x.lower() else negative_class)

    # label mapping to 0 and 1 for non-news and news respectively.
    df['label'] = df[class_column].map(label_map)

    # Change the column names to be more descriptive. And drop the original columns.
    # This is useful when we want to use the same code for other datasets.
    # TODO make this more generic.
    df['text'] = df['headline'] + ' ' + df['short_description']
    df = df[['label', 'text']]

    # Add word count column. Useful for choosing the max_seq_length.
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    # Drop duplicates in the text column
    df = df.drop_duplicates(subset='text', keep='first')

    # Preprocess the text might not be necessary for BERT.
    #   https://stackoverflow.com/questions/70649831/does-bert-model-need-text
    #   https://datascience.stackexchange.com/questions/113359/why-there-is-no-preprocessing-step-for-training-bert
    # df['text'] = df['text'].apply(clean_text)

    return df


def get_dataset_df(data_path):
    """Get the dataset dataframe."""
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f'Unknown file format: {data_path}')
    return df


def create_datasets(config, device, output_dir_path):
    df_multi_class = get_dataset_df(data_path=config.data.data_path)

    # label mapping to 0 and 1 for negative and positive respectively.
    df = preprocess_dataframe(df=df_multi_class.copy(), config=config)

    # Split the dataframe into training, test, and validation sets
    train_df = df.sample(frac=config.data.train_size, random_state=SEED)

    # Split the text into segments of seq_length tokens and overlap of 50 tokens.
    if config.data.split_text is not None:
        train_df['text'] = train_df['text'].apply(partial(create_segments,
                                                          segment_length=config.model.max_seq_length,
                                                          overlap=config.data.split_text.overlap_size))
        train_df = train_df.explode('text')

    val_df = df.drop(train_df.index).sample(frac=config.data.val_size, random_state=SEED)
    test_df = df.drop(train_df.index).drop(val_df.index)

    sampler = get_sampler(train_df, config)

    tokenizer = BertTokenizer.from_pretrained(config.model.model_name, do_lower_case=config.model.uncased)
    train_dr = get_dataloader(tokenizer=tokenizer, df=train_df, config=config, device=device, sampler=sampler)
    val_dr = get_dataloader(tokenizer=tokenizer, df=val_df, config=config, device=device)
    test_dr = get_dataloader(tokenizer=tokenizer, df=test_df, config=config, device=device)

    if config.data.plot_histograms:
        plots_dir = os.path.join(output_dir_path, 'data_plots')
        os.makedirs(plots_dir, exist_ok=True)

        plot_column_histogram(df_multi_class, column=config.data.class_column,
                              title=f'Dataset Histogram (Multi-class)',
                              output_dir_path=plots_dir)
        plot_column_histogram(df, column='label',
                              title=f'Dataset Histogram',
                              output_dir_path=plots_dir)
        for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
            plot_column_histogram(df, column='label',
                                  title=f'{name.upper()} Histogram',
                                  output_dir_path=plots_dir)
            if name == 'train':
                plot_column_histogram(df, column='word_count',
                                      title=f'{name.upper()} Word Count Histogram',
                                      x_label='Number of Words', y_label='Frequency',
                                      add_text_to_bars_and_ticks=False,
                                      output_dir_path=plots_dir)
            print(f'[INFO] {name} set size: {len(df)}')

    return train_dr, val_dr, test_dr

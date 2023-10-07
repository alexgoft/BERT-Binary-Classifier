import torch

import string
import re
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils import plot_column_histogram
from nltk.corpus import stopwords
cached_stop_words = stopwords.words("english")

CONTENT_CLASS_COLUMN = 'content_type'
POSITIVE_STR = 'news'
NEGATIVE_STR = 'non-news'
LABEL_MAP = {POSITIVE_STR: 1, NEGATIVE_STR: 0}

class TextDataset(Dataset):

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

        # Column names for the data and the labels
        self.data_col = data_col
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row[self.data_col]
        label = data_row[self.label_col]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
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


def get_dataloader(tokenizer, df, config, device=torch.device('cpu')):
    """Create dataloader for the given dataframe."""
    ds = TextDataset(df, tokenizer, max_token_len=config.data.max_seq_length, device=device)
    dr = DataLoader(ds, batch_size=config.train.batch_size, shuffle=True)
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
    text = ' '.join([word for word in text.split() if word not in cached_stop_words])
    return text


def preprocess_dataframe(df):
    """
    Preprocess the dataframe. This includes:
        - label mapping to 0 and 1 for non-news and news respectively.
        - Change the column names to be more descriptive. And drop the original columns.
        - Preprocess the text (lowercase, remove punctuation, remove stopwords).
    """
    # label mapping to 0 and 1 for non-news and news respectively.
    df['label'] = df[CONTENT_CLASS_COLUMN].map(LABEL_MAP)

    # Change the column names to be more descriptive. And drop the original columns.
    # This is useful when we want to use the same code for other datasets.
    df['text'] = df['scraped_title'] + ' ' + df['scraped_text']
    df = df[['label', 'text']]

    # Preprocess the text
    df['text'] = df['text'].apply(clean_text)

    return df


def create_datasets(config, device, plot_histograms=False):

    # Read the data and plot the histogram of the content type column.
    df = pd.read_csv(config.data.data_path)

    # Binaries the content type column and plot the histogram again.
    df[CONTENT_CLASS_COLUMN] = df[CONTENT_CLASS_COLUMN].apply(
        lambda x: POSITIVE_STR if x == POSITIVE_STR else NEGATIVE_STR)

    # label mapping to 0 and 1 for non-news and news respectively.
    df = preprocess_dataframe(df)

    # Split the dataframe into training, test, and validation sets
    seed = config.data.random_seed
    train_df = df.sample(frac=config.data.train_size, random_state=seed)
    val_df = df.drop(train_df.index).sample(frac=config.data.val_size, random_state=seed)
    test_df = df.drop(train_df.index).drop(val_df.index)

    tokenizer = BertTokenizer.from_pretrained(config.model.model_name,
                                              do_lower_case=config.model.uncased)
    train_dr = get_dataloader(tokenizer=tokenizer, df=train_df, config=config, device=device)
    val_dr = get_dataloader(tokenizer=tokenizer, df=val_df, config=config, device=device)
    test_dr = get_dataloader(tokenizer=tokenizer, df=test_df, config=config, device=device)

    if plot_histograms:
        for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
            plot_column_histogram(df, column='label', title=f'Training set histogram {name}')
            print(f'[INFO] {name} set size: {len(test_df)}')

    return train_dr, val_dr, test_dr

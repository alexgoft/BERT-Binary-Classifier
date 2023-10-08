import torch
import pandas as pd

from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from samplers import get_sampler
from text_dataset import TextDataset
from plot_utils import plot_column_histogram

CACHED_STOP_WORDS = stopwords.words("english")

CONTENT_CLASS_COLUMN = 'content_type'
POSITIVE_STR = 'news'
NEGATIVE_STR = 'non-news'
LABEL_MAP = {POSITIVE_STR: 1, NEGATIVE_STR: 0}


def get_dataloader(tokenizer, df, config, device=torch.device('cpu'), sampler=None):
    """Create dataloader for the given dataframe."""
    ds = TextDataset(df, tokenizer, max_token_len=config.data.max_seq_length, device=device)
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
    # Preprocess the text might not be necessary for BERT.
    # https://stackoverflow.com/questions/70649831/does-bert-model-need-text
    # https://datascience.stackexchange.com/questions/113359/why-there-is-no-preprocessing-step-for-training-bert

    # text = text.replace('\n', ' ')
    # text = re.sub(r'\d+', '', text)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # text = ' '.join([word for word in text.split() if word not in CACHED_STOP_WORDS])
    return text


def preprocess_dataframe(df):
    """
    Preprocess the dataframe. This includes:
        - label mapping to 0 and 1 for non-news and news respectively.
        - Change the column names to be more descriptive. And drop the original columns.
        - Preprocess the text (lowercase, remove punctuation, remove stopwords).
    """
    # Binaries the content type column and plot the histogram again.
    df[CONTENT_CLASS_COLUMN] = df[CONTENT_CLASS_COLUMN].apply(
        lambda x: POSITIVE_STR if x == POSITIVE_STR else NEGATIVE_STR)

    # label mapping to 0 and 1 for non-news and news respectively.
    df['label'] = df[CONTENT_CLASS_COLUMN].map(LABEL_MAP)

    # Change the column names to be more descriptive. And drop the original columns.
    # This is useful when we want to use the same code for other datasets.
    df['text'] = df['scraped_title'] + ' ' + df['scraped_text']
    df = df[['label', 'text']]

    # Preprocess the text
    df['text'] = df['text'].apply(clean_text)

    return df


def create_datasets(config, device):
    # Read the data_utils and plot the histogram of the content type column.
    df = pd.read_csv(config.data.data_path)

    # label mapping to 0 and 1 for non-news and news respectively.
    df = preprocess_dataframe(df)

    # Split the dataframe into training, test, and validation sets
    train_df = df.sample(frac=config.data.train_size, random_state=config.general.seed)
    val_df = df.drop(train_df.index).sample(frac=config.data.val_size, random_state=config.general.seed)
    test_df = df.drop(train_df.index).drop(val_df.index)

    sampler = get_sampler(train_df, config)

    tokenizer = BertTokenizer.from_pretrained(config.model.model_name, do_lower_case=config.model.uncased)
    train_dr = get_dataloader(tokenizer=tokenizer, df=train_df, config=config, device=device, sampler=sampler)
    val_dr = get_dataloader(tokenizer=tokenizer, df=val_df, config=config, device=device)
    test_dr = get_dataloader(tokenizer=tokenizer, df=test_df, config=config, device=device)

    if config.data.plot_histograms:
        for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
            plot_column_histogram(df, column='label', title=f'Training set histogram {name}')
            print(f'[INFO] {name} set size: {len(test_df)}')

    return train_dr, val_dr, test_dr
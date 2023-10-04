import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils import plot_column_histogram

DATA_PATH = 'assignment_data_en.csv'
CONTENT_CLASS_COLUMN = 'content_type'
POSITIVE_STR = 'news'
NEGATIVE_STR = 'non-news'
LABEL_MAP = {POSITIVE_STR: 1, NEGATIVE_STR: 0}

TRAIN_SET_SIZE = 0.75
VALIDATION_SET_SIZE = 0.3

MAX_TOKEN_LEN = 256
RANDOM_SEED = 42


class TextDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = 128,
            label_col: str = 'label',
            data_col: str = 'title_text',
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
        label_vec = np.zeros(shape=(len(LABEL_MAP)), dtype=np.float32)
        label_vec[label] = 1.0  # one-hot encoding of the label. 0->non-news, 1->news
        label_vec = torch.FloatTensor(label_vec)

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
            comment_text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=label_vec
        )


def get_dataloader(tokenizer, df, batch_size, device):
    ds = TextDataset(df, tokenizer, max_token_len=MAX_TOKEN_LEN, device=device)
    dr = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dr


def create_datasets(batch_size, device):
    # Read the data and plot the histogram of the content type column.
    df = pd.read_csv(DATA_PATH)

    # Binaries the content type column and plot the histogram again.
    df[CONTENT_CLASS_COLUMN] = df[CONTENT_CLASS_COLUMN].apply(
        lambda x: POSITIVE_STR if x == POSITIVE_STR else NEGATIVE_STR)

    # label mapping to 0 and 1 for non-news and news respectively.
    df['label'] = df[CONTENT_CLASS_COLUMN].map(LABEL_MAP)

    # Keep only the label and the text.
    # Create column with the title and the text and drop the rest of the columns.
    df['title'] = df['scraped_title']
    df['text'] = df['scraped_text']
    df['title_text'] = df['text']  # df['title'] + ' ' + df['text']
    df = df[['label', 'title', 'text', 'title_text']]

    # TODO Clean text column. Remove punctuation, stopwords, etc. See:
    #  https://www.kaggle.com/parulpandey/eda-and-preprocessing-for-bert
    #  https://towardsdatascience.com/nlp-in-python-data-cleaning-6313a404a470
    #  https://towardsdatascience.com/how-to-preprocess-text-data-using-nlp-tools-9f6dcab5ccb9
    #  https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/

    # Split the dataframe into training, test, and validation sets
    train_df = df.sample(frac=TRAIN_SET_SIZE, random_state=RANDOM_SEED)
    val_df = df.drop(train_df.index).sample(frac=VALIDATION_SET_SIZE, random_state=RANDOM_SEED)
    test_df = df.drop(train_df.index).drop(val_df.index)

    # plot_column_histogram(df, column=CONTENT_CLASS_COLUMN, title='Content type histogram')
    # plot_column_histogram(train_df, column='label', title='Training set histogram TRAIN')
    # plot_column_histogram(val_df, column='label', title='Training set histogram VAL')
    # plot_column_histogram(test_df, column='label', title='Training set histogram TEST')

    print('[INFO] Train set size: {}'.format(len(train_df)))
    print('[INFO] Validation set size: {}'.format(len(val_df)))
    print('[INFO] Test set size: {}'.format(len(test_df)))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_dr = get_dataloader(tokenizer, train_df, batch_size, device)
    val_dr = get_dataloader(tokenizer, val_df, batch_size, device)
    test_dr = get_dataloader(tokenizer, test_df, batch_size, device)

    return train_dr, val_dr, test_dr

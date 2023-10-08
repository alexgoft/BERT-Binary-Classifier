import torch
import pandas as pd
from transformers import BertTokenizer

from torch.utils.data import Dataset


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

        # Column names for the data_utils and the labels
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

import torch

from torch import nn
from transformers import BertModel


class NewsClassifier(nn.Module):
    BERT_MODEL_NAME = 'bert-base-uncased' # 'bert-base-cased'

    def __init__(self, n_classes=2, device=torch.device('cpu')):
        super(NewsClassifier, self).__init__()
        # self._n_classes = n_classes  # TODO Try with 1 class?
        self._n_classes = 1

        # Bert is a transformer model that is pretrained on a large corpus of text.
        # We will use the BERT model to extract meaningful representations of our text
        # that we will feed to a classifier. We will use the pre-trained weights of the
        # BERT model.
        self.bert = BertModel.from_pretrained(self.BERT_MODEL_NAME, return_dict=True)
        # for name, param in self.bert.named_parameters():  # TODO Freeze BERT weights?
        #     param.requires_grad = False

        # ---------------------------------------------------------------------
        self.classifier = nn.Linear(self.bert.config.hidden_size, self._n_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.criterion = nn.BCELoss()

        self.device = device
        self.to(device)

    def forward(self, batch):
        attention_mask = batch["attention_mask"].to(device=self.device)
        text_tokenized = batch["text_tokenized"].to(device=self.device)
        labels = None
        if "label" in batch:
            labels = batch["label"].to(device=self.device)
            # labels = torch.Tensor(labels).long()
            # labels = torch.nn.functional.one_hot(labels, num_classes=self._n_classes).to(torch.float32)
            labels = labels.unsqueeze(1).to(torch.float32)  # TODO Try with 1 class?

        output = self.bert(text_tokenized, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)
        output = torch.sigmoid(output)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

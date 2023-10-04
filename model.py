import torch

from torch import nn
from transformers import BertModel, BertForSequenceClassification, AutoConfig


class NewsClassifier(nn.Module):
    BERT_MODEL_NAME = 'bert-base-cased'

    def __init__(self, n_classes=2, device=torch.device('cpu')):
        super(NewsClassifier, self).__init__()

        # TODO Use BertForSequenceClassification instead?
        # self.bert = BertForSequenceClassification.from_pretrained(self.BERT_MODEL_NAME, return_dict=True)

        # TODO Use BertModel instead?
        # Bert is a transformer model that is pretrained on a large corpus of text.
        # We will use the BERT model to extract meaningful representations of our text
        # that we will feed to a classifier. We will use the pre-trained weights of the
        # BERT model.
        # self.bert = BertModel.from_pretrained(self.BERT_MODEL_NAME, return_dict=True)
        # TODO Freeze BERT weights?
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        # Load the pre-trained BERT model configuration
        configuration = AutoConfig.from_pretrained('bert-base-uncased')
        configuration.hidden_dropout_prob = 0.1
        configuration.attention_probs_dropout_prob = 0.1

        # Instantiate the BERT model with the customized configuration
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
                                              config=configuration)

        # ---------------------------------------------------------------------
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = nn.BCELoss()

        self.device = device
        self.to(device)

    def forward(self, batch):
        input_ids = batch["input_ids"].to(device=self.device)
        attention_mask = batch["attention_mask"].to(device=self.device)
        labels = batch["label"].to(device=self.device) if "label" in batch else None

        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

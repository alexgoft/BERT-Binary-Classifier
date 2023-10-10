import torch

from torch import nn
from transformers import BertModel


class BERTNewsClassifier(nn.Module):
    """
        Bert is a transformer model that is pretrained on a large corpus of text.
        We will use the BERT model to extract meaningful representations of our text
        that we will feed to a classifier. We will use the pre-trained weights of the
        BERT model.
    """
    def __init__(self, config, device=torch.device('cpu')):
        super(BERTNewsClassifier, self).__init__()

        self._n_classes = config.model.n_classes
        self._dropout = nn.Dropout(p=config.train.dropout)
        self._loss_function = nn.BCELoss()

        self._initialize_bert(config)
        self._initialize_classification_layer(config)

        self._device = device
        self.to(device)

    def _initialize_bert(self, config):
        """Initialize the BERT model. Load the pre-trained weights from transformers library."""
        self._bert = BertModel.from_pretrained(config.model.model_name, return_dict=True)
        if config.model.freeze_bert:
            for param in self._bert.parameters():
                param.requires_grad = False

    def _initialize_classification_layer(self, config):
        """Initialize the classification layer.
           It is a linear layer that takes the output of the BERT model."""
        if config.model.linear_layers_num == 1:
            self._classifier = nn.Linear(self._bert.config.hidden_size, self._n_classes)
        elif config.model.linear_layers_num == 2:
            self._classifier = nn.Sequential(
                nn.Linear(self._bert.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self._n_classes)
            )
        else:
            raise ValueError("linear_layers_num must be 1 or 2")

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def forward(self, batch):
        attention_mask = batch["attention_mask"].to(device=self._device)
        text_tokenized = batch["text_tokenized"].to(device=self._device)
        labels = None
        if "label" in batch:
            labels = batch["label"].to(device=self._device)
            # labels = torch.Tensor(labels).long()
            # labels = torch.nn.functional.one_hot(labels, num_classes=self._n_classes).to(torch.float32)
            labels = labels.unsqueeze(1).to(torch.float32)  # TODO Try with 1 class?

        output = self._bert(text_tokenized, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        pooled_output = self._dropout(pooled_output)
        output = self._classifier(pooled_output)
        output = torch.sigmoid(output)

        loss = 0
        if labels is not None:
            loss = self._loss_function(output, labels)
        return loss, output

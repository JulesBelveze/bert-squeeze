import torch
import torch.nn as nn

from .mha import MultiHeadedAttention


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.input_size = config.input_size
        self.classifier_hidden_size = 128
        self.classifier_heads_num = 2
        self.num_labels = config.num_labels
        self.pooling = config.pooling

        self.fc = nn.Linear(config.input_size, self.classifier_hidden_size)
        self.self_att = MultiHeadedAttention(
            self.classifier_hidden_size, self.classifier_heads_num, config.dropout_prob
        )
        self.pre_classifier = nn.Linear(
            self.classifier_hidden_size, self.classifier_hidden_size
        )
        self.classifier = nn.Linear(self.cla_hidden_size, config.num_labels)

    def forward(self, hidden, mask):
        """"""
        hidden = torch.tanh(self.fc(hidden))
        hidden = self.self_att(hidden, hidden, hidden, mask)

        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]

        pre_output = torch.tanh(self.pre_classifier(hidden))
        logits = self.classifier(pre_output)
        return logits

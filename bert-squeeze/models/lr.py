import torch
import torch.nn.functional as F


class BowLogisticRegression(torch.nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, **kwargs):
        super(BowLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(vocab_size, num_labels)

    def forward(self, features, **kwargs):
        return F.log_softmax(self.linear(features), dim=1)


class EmbeddingLogisticRegression(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_labels: int, mode: str = "mean", **kwargs):
        super(EmbeddingLogisticRegression, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, mode=mode)
        self.classifier = torch.nn.Linear(embed_dim, num_labels)

    def forward(self, features, offsets=None, **kwargs):
        # for some reason tensors are on CPU, need to move them
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedded = self.embedding(features.to(device), offsets)
        return self.classifier(embedded)

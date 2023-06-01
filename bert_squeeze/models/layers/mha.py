# Taken from:
# https://github.com/autoliuweijie/FastBERT/blob/859632f67eb97b1624b26c8f8766972153e6382b/uer/layers/multi_headed_attn.py

import math

import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()

        def unshape(x):
            return (
                x.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
            )

        query, key, value = [
            layer(x)
            .view(batch_size, -1, self.heads_num, self.per_head_size)
            .transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores /= math.sqrt(float(self.per_head_size))
        scores += mask
        probs = torch.nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        return self.final_linear(output)

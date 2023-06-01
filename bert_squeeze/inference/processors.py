from typing import List, Union

import torch
from transformers import AutoTokenizer


class BasicPreprocessor(object):
    """
    BasicPreProcessor
    """

    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, text: Union[str, List[str]], *args, **kwargs):
        return self.tokenizer(text, return_tensors="np")


class BasicPostProcessor(object):
    """BasicPostProcessor"""

    def __init__(self):
        pass

    def __call__(self, model_output: torch.Tensor, *args, **kwargs):
        probs = torch.softmax(model_output, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds

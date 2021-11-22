import torch
from .bert import BertCustomEncoder
from transformers.models.bert import BertModel


class CustomLabseModel(BertModel):
    """
    CustomLabse model using the `BertCustomEncoder` to be able to perform layer dropping.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = BertCustomEncoder(config)

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Taken from here: https://github.com/huggingface/transformers/blob/fd8136fa755a3c59e459e1168014f2bf2fca721a/src/transformers/modeling_utils.py#L488
        to avoid downloading pretrained checkpoints every time.
        All context managers that the model should be initialized under go here.
        Args:
            torch_dtype (:obj:`torch.dtype`, `optional`):
                Override the default ``torch.dtype`` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

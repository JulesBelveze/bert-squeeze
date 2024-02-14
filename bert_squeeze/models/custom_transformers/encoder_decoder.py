from typing import Union

import torch
import torch.nn as nn
from transformers import AutoModel, VisionEncoderDecoderModel


class BaseEncoderDecoderModel(nn.Module):
    """
    A model that contains an encoder and a decoder, with support for loading from pre-trained transformer checkpoints.
    """

    BASE_MODEL_CLASS = AutoModel

    def __init__(
        self,
        model: Union[str, nn.Module] = None,
        encoder: Union[str, nn.Module] = None,
        decoder: Union[str, nn.Module] = None,
        *args,
        **kwargs,
    ):
        """Initialize the EncoderDecoderModel with either a single model or separate encoder and decoder.

        Args:
            model (Union[str, nn.Module]): A string specifying the pre-trained model or an instance of a model
                                           with encoder and decoder.
            encoder (Union[str, nn.Module]): A string specifying the pre-trained encoder model or an instance
                                             of an encoder.
            decoder (Union[str, nn.Module]): A string specifying the pre-trained decoder model or an instance
                                             of a decoder.
        """
        super().__init__(*args, **kwargs)
        assert model is not None or (
            encoder is not None and decoder is not None
        ), "You need to provide either a model or an encoder and a decoder"

        if model is not None:
            if isinstance(model, str):
                model = self.BASE_MODEL_CLASS.from_pretrained(model)

            assert hasattr(model, "encoder") and hasattr(
                model, "decoder"
            ), "The model you provide must contain an encoder and a decoder module"
            self.model = model
        else:
            if isinstance(encoder, str):
                encoder = self.BASE_MODEL_CLASS.from_pretrained(encoder)
            if isinstance(decoder, str):
                decoder = self.BASE_MODEL_CLASS.from_pretrained(decoder)

            self.model = nn.ModuleDict({"encoder": encoder, "decoder": decoder})

    @property
    def encoder(self) -> nn.Module:
        """Returns the current encoder of the model."""
        return self.model.encoder

    @encoder.setter
    def encoder(self, encoder: Union[str, nn.Module]) -> None:
        """Sets a new encoder for the model.

        Args:
            encoder (Union[str, nn.Module]): A string specifying the pre-trained model or an instantiated model.
        """
        self.replace_encoder(encoder)

    @property
    def decoder(self) -> nn.Module:
        """Returns the current decoder of the model."""
        return self.model.decoder

    @decoder.setter
    def decoder(self, decoder: Union[str, nn.Module]) -> None:
        """Sets a new decoder for the model.

        Args:
            decoder (Union[str, nn.Module]): A string specifying the pre-trained model or an instantiated model.
        """
        self.replace_decoder(decoder)

    def replace_encoder(self, model: Union[str, nn.Module]) -> None:
        """Replace the current encoder of the model with a new one.

        Args:
            model (Union[str, nn.Module]): A string specifying the pre-trained encoder model or an instance of a
                                           model with an encoder module.
        """
        if isinstance(model, str):
            encoder = AutoModel.from_pretrained(model)
        else:
            if hasattr(model, "encoder"):
                encoder = model.encoder
            else:
                encoder = model
        self.model.encoder = encoder

    def replace_decoder(self, model: Union[str, nn.Module]) -> None:
        """Replace the current decoder of the model with a new one.

        Args:
            model (Union[str, nn.Module]): A string specifying the pre-trained decoder model or an instance of a
                                           model with a decoder module.
        """
        if isinstance(model, str):
            decoder = AutoModel.from_pretrained(model)
        else:
            if hasattr(model, "decoder"):
                decoder = model.decoder
            else:
                decoder = model
        self.model.decoder = decoder

    def forward(self, *args, **kwargs):
        """"""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """"""
        return self.model.generate(*args, **kwargs)


class VisionEncoderDecoder(BaseEncoderDecoderModel):
    """"""

    BASE_MODEL_CLASS = VisionEncoderDecoderModel

    def __init__(
        self,
        model: Union[str, nn.Module] = None,
        encoder: Union[str, nn.Module] = None,
        decoder: Union[str, nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model, encoder, decoder, *args, **kwargs)

    def forward(self, pixel_values: torch.Tensor, *args, **kwargs):
        """"""
        return self.model(pixel_values)

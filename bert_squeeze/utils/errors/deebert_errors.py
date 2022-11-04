from ...models.custom_transformers.deebert import RampOutput
from typing import Tuple


class RampException(Exception):
    """
    This error is only used to exit the encoder stack earlier.

    Args:
        message (Tuple[RampOutput]):
            off-ramp outputs collected until model exit
        exit_layer (int):
            index of the exit layer
    """

    def __init__(self, message: Tuple[RampOutput], exit_layer: int):
        self.message = message
        self.exit_layer = exit_layer

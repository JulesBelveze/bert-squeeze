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

    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer

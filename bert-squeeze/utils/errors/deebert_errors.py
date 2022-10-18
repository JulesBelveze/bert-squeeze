class RampException(Exception):
    """
    This error is only used to exit the encoder stack earlier.

    Args:
        message (str):
            error message
        exit_layer (int):
            index of the exit layer
    """

    def __init__(self, message: str, exit_layer: int):
        self.message = message
        self.exit_layer = exit_layer

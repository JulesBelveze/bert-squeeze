class RampException(Exception):
    """
    This error is only used to exit the encoder stack earlier.
    """

    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer

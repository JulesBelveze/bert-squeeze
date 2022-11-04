class ConfigurationException(Exception):
    def __init__(self, message="Configuration error."):
        self.message = message
        super().__init__(self.message)

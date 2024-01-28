from transformers import T5ForConditionalGeneration, T5Config
from bert_squeeze.utils.callbacks.pruning import LayerPruning


class PLModule:
    def __init__(self, model):
        self.student = model


def test_layer_pruning():
    cb = LayerPruning(3, 3)

    config = T5Config()
    model = T5ForConditionalGeneration(config)
    assert model.config.num_layers == 6
    assert model.config.num_decoder_layers == 6

    trainer = None
    pl_module = PLModule(model)

    cb.setup(trainer, pl_module, "fit")

    assert model.config.num_layers == 3
    assert model.config.num_decoder_layers == 3

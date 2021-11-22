import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning


class FastBertLogic(BaseFinetuning):
    """
    Callback freezing the model backbone after 'freeze_at_epoch' epochs.
    """

    def __init__(self, freeze_at_epoch=10):
        self.freeze_at_epoch = freeze_at_epoch

    def freeze_before_training(self, pl_module: pl.LightningModule):
        """"""
        assert pl_module.training_stage == 0, "The 'training_stage' should be 0 when starting to finetune FastBert."

    def finetune_function(self, pl_module: pl.LightningModule, current_epoch, optimizer, optimizer_idx):
        """"""
        if current_epoch >= self.freeze_at_epoch:
            pl_module.training_stage = 1
            pl_module.freeze_encoder()

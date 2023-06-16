import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer


class FastBertLogic(BaseFinetuning):
    """
    Callback freezing the model backbone after 'freeze_at_epoch' epochs.

    The `finetune_function` method is called on every train epoch start and should
    be used to `unfreeze` any parameters. Those parameters need to be added in a new
    `param_group` within the optimizer.

    Args:
        freeze_at_epoch (int):
            number of epochs after which to freeze the model's backbone
    """

    def __init__(self, freeze_at_epoch: int = 10) -> None:
        super().__init__()
        self.freeze_at_epoch = freeze_at_epoch

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """

        Args:
            pl_module:
        """
        assert (
            pl_module.training_stage == 0
        ), "The 'training_stage' should be 0 when starting to finetune FastBert."

    def finetune_function(
        self, pl_module: pl.LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        """
        Called when the epoch begins

        Args:
            pl_module (pl.LightningModule):
            current_epoch (int):
                current epoch we are in
            optimizer (Optimizer):
                optimizer used for training
        """
        if current_epoch >= self.freeze_at_epoch:
            pl_module.training_stage = 1
            pl_module.freeze_encoder()

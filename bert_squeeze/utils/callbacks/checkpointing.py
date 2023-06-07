##########################################
########## Code from andrewjong ##########
##########################################
import os

import lightning.pytorch as pl


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.

    Args:
        save_step_frequency (int):
            how often to save in steps
        prefix (str):
            add a prefix to the name, only used when `use_model_checkpoint_filename=False`
        use_model_checkpoint_filename (bool):
            just default filename, don't use ours.
    """

    def __init__(
        self,
        save_step_frequency: int,
        prefix: str = "N-Step-Checkpoint",
        use_model_checkpoint_filename: bool = False,
    ) -> None:
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_model_checkpoint_filename = use_model_checkpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _) -> None:
        """
        Check if we should save a checkpoint after every train batch

        Args:
            trainer (pl.Trainer)
        """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_model_checkpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

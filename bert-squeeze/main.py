# To run such a program one can run it the following way:
# python3 -m bert-squeeze.main -cp=configs -cn=training_config
#
# To override arguments of the config file run as follow:
# python3 -m bert-squeeze.main -cp=configs -cn=training_config --task=test +new_attr=test

import logging
import sys

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from pkg_resources import resource_filename
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from .utils import load_model_from_exp, get_neptune_tags
from .utils.callbacks import CheckpointEveryNSteps
from .utils.errors import ConfigurationException

load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@hydra.main(config_path="./configs/", config_name="training_config")
def run(args):
    logging.info(f"Using config: {args}")

    data = instantiate(args.data)
    data.prepare_data()
    data.setup()

    if args.general.do_train:
        neptune_logger = instantiate(args.neptune.logger)
        neptune_logger.experiment["sys/tags"].add(get_neptune_tags(args))
        neptune_logger.log_hyperparams(args)

        model = instantiate(args.model, _recursive_=False)

        callbacks = [CheckpointEveryNSteps(args.general.save_steps)]
        if args.train.get("lr_scheduler", False):
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))

        if args.general.get("quantization", None) is not None:
            quantization_callback = instantiate(args.general.quantization)
            callbacks.append(quantization_callback)

        if args.general.get("pruning", None) is not None:
            pruning_callback = instantiate(args.general.pruning)
            callbacks.append(pruning_callback)

        if "fast" in args.model._target_:
            callbacks.append(instantiate(args.finetuning_callback))

        # NOTE: when performing manual optimization the 'gradient_clip_val' flag needs
        # to be set to None.
        # Issue here: https://github.com/PyTorchLightning/pytorch-lightning/issues/7698
        trainer = Trainer(
            gpus=torch.cuda.device_count(),
            accumulate_grad_batches=args.train.accumulation_steps,
            gradient_clip_val=args.train.max_grad_norm,
            accelerator='ddp',
            auto_lr_find=args.train.auto_lr,
            logger=neptune_logger,
            callbacks=callbacks,
            check_val_every_n_epoch=args.general.validation_every_n_epoch
        )

        logging.info(f"Starting training: {model}")

        trainer.fit(
            model=model,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader()
        )

        # exporting trained model to ONNX
        input_sample = iter(data.test_dataloader).next()
        model.to_onnx(f"{args.general.output_dir}/model.onnx", input_sample, export_params=True)

    if args.general.do_eval:
        if not hasattr(args.general, "model_path"):
            raise ConfigurationException("You are on 'eval' mode you need to specify path to model checkpoint.")
        args.general.model_path = resource_filename("bert-squeeze", args.general.model_path)

        model = load_model_from_exp(path_to_folder=args.general.model_path, module=args.model._target_)

        model.eval()
        trainer = Trainer(
            gpus=torch.cuda.device_count(),
            accelerator='ddp'
        )
        trainer.test(model, datamodule=data)


if __name__ == "__main__":
    run()

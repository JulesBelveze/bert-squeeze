import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import datasets
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class HardLabeler(object):
    """
    Helper object that will hard label a dataset using a model's predictions.

    This is useful when lacking annotated data and performing model distillation. This
    way we generate an annotated dataset from an unlabelled one.

    The HardLabeler will load a pretrained a model as well as a dataset. It will then
    tokenize the dataset and add a "labels" key to the features' dictionary.

    NOTE: only the train dataset will be hard labeled.

    Args:
        labeler_config (DictConfig):
            configuration of the model to load
        dataset_config (DictConfig):
            configuration of the dataset to load
        max_length (int):
            maximum sequence length
    """

    def __init__(
        self,
        labeler_config: DictConfig,
        dataset_config: DictConfig,
        max_length: int,
        **kwargs,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, tokenizer = self._get_model(labeler_config)
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.dataset_config = dataset_config
        self.max_length = max_length
        self.batch_size = kwargs.get("train_batch_size", 32)

    @staticmethod
    def _get_model(config: DictConfig) -> Tuple[pl.LightningModule, AutoTokenizer]:
        """
        Loads the model checkpoints and the tokenizer.

        Args:
            config (DictConfig):
                configuration of the model to load
        Returns:
            Tuple[pl.LightningModule, AutoTokenizer]:
                fine-tuned model along with the associated tokenizer
        """
        if config.get("checkpoint_path") is not None:
            checkpoint_path = resource_filename("bert_squeeze", config.checkpoint_path)
            teacher_class = config.teacher

            teacher = teacher_class.load_from_checkpoint(
                checkpoint_path,
                training_config=config,
                pretrained_model=config.pretrained_model,
                num_labels=config.num_labels,
            )
        else:
            teacher = config.teacher

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model, model_max_len=config.max_length
        )
        return teacher, tokenizer

    def featurize(
        self, dataset: Union[DatasetDict, Dataset]
    ) -> Union[DatasetDict, Dataset]:
        """
        Args:
            dataset (datasets.DatasetDict):
                dataset to featurize
        Returns:
            DatasetDict: featurized dataset
        """
        tokenized_dataset = dataset.map(
            lambda x: self.tokenizer(
                x[self.dataset_config.text_col],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        )
        # removing extra columns
        columns_to_remove = list(
            set(tokenized_dataset.column_names)
            - {"input_ids", "token_type_ids", "attention_mask"}
        )
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        return tokenized_dataset

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        """"""
        if self.dataset_config.is_local:
            dataset = datasets.load_dataset(
                resource_filename(
                    "bert-squeeze", f"data/{self.dataset_config.path}_dataset.py"
                ),
                self.dataset_config.split,
            )
        else:
            dataset = datasets.load_dataset(
                self.dataset_config.path, self.dataset_config.split
            )
        dataset = dataset["train"].select(range(self.dataset_config.max_samples))

        logging.info("Featurizing hard dataset for labeling.")
        featurized_dataset = self.featurize(dataset)

        featurized_dataset.set_format(type="pt")
        return DataLoader(featurized_dataset, batch_size=self.batch_size, drop_last=True)

    def label_dataset(self) -> Dict[str, List[Any]]:
        """
        Annotates the unlabeled dataset using the fine-tuned teacher.

        Returns:
            Dict[str, List[Any]]:
                dictionary containing features and hard labels
        """
        output_data = defaultdict(list)
        train_loader = self.get_dataloader()

        self.model.eval()
        for batch in tqdm(train_loader, total=len(train_loader), desc="Labeling samples"):
            with torch.no_grad():
                logits = self.model(
                    **{k: v.to(self.device) for k, v in batch.items()}
                ).logits
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                output_data["labels"].extend(preds.tolist())

                for col in ["input_ids", "token_type_ids", "attention_mask"]:
                    output_data[col].extend(batch[col].tolist())

        return output_data

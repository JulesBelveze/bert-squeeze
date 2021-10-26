import logging
from collections import defaultdict

import datasets
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class HardLabeler(object):
    def __init__(self, labeler_config: DictConfig, dataset_config: DictConfig, max_length: int, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, tokenizer = self._get_model(labeler_config)
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.dataset_config = dataset_config
        self.max_length = max_length
        self.batch_size = kwargs.get("train_batch_size", 32)

    def _get_model(self, config: DictConfig):
        """"""
        checkpoint_path = resource_filename("bert-squeeze", config.checkpoint_path)
        teacher_class = config.teacher

        teacher = teacher_class.load_from_checkpoint(
            checkpoint_path,
            training_config=config,
            pretrained_model=config.pretrained_model,
            num_labels=config.num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, model_max_len=config.max_length)
        return teacher, tokenizer

    def featurize(self, dataset):
        """"""
        tokenized_dataset = dataset.map(
            lambda x: self.tokenizer(x[self.dataset_config.text_col], padding="max_length", max_length=self.max_length,
                                     truncation=True)
        )
        # removing extra columns
        columns_to_remove = list(
            set(tokenized_dataset.column_names) - {"input_ids", "token_type_ids", "attention_mask"}
        )
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        tokenized_dataset = tokenized_dataset.add_column("id", range(tokenized_dataset.num_rows))
        return tokenized_dataset

    def get_dataloader(self):
        """"""

        def _collate_fn():
            def _collate(examples):
                return self.tokenizer.pad(examples, return_tensors="pt")

            return _collate

        if self.dataset_config.is_local:
            dataset = datasets.load_dataset(
                resource_filename("bert-squeeze", f"data/{self.dataset_config.name}_dataset.py"),
                self.dataset_config.split
            )
        else:
            dataset = datasets.load_dataset(self.dataset_config.name, self.dataset_config.split)
        dataset = dataset["train"].select(range(self.dataset_config.max_samples))

        logging.info("Featurizing hard dataset for labeling.")
        featurized_dataset = self.featurize(dataset)

        featurized_dataset.set_format(type='torch', columns=["input_ids", "token_type_ids", "attention_mask", "id"])
        return DataLoader(featurized_dataset, collate_fn=_collate_fn(), batch_size=self.batch_size,
                          drop_last=True, num_workers=0)

    def label_dataset(self):
        """
        Annotating the unlabeled dataset using the fine tuned teacher. We return a dictionary
        mapping the document id to its corresponding label.
        """
        output_data = defaultdict(list)
        train_loader = self.get_dataloader()

        self.model.eval()
        for batch in tqdm(train_loader, total=len(train_loader), desc="Labeling samples"):
            with torch.no_grad():
                logits = self.model(**batch.to(self.device))
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(axis=-1)

                output_data["labels"].extend(preds.tolist())

                for col in ["input_ids", "token_type_ids", "attention_mask"]:
                    output_data[col].extend(batch[col].tolist())

        return output_data

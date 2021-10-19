import torch.nn.functional as F
import datasets
import pkg_resources
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


class HardLabeler(object):
    def __init__(self, model, dataset_config, tokenizer_name, max_length, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)

        self.dataset_config = dataset_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_len=max_length
        )
        self.max_length = max_length
        self.batch_size = kwargs.get("train_batch_size", 32)

    def get_dataloader(self):
        def _collate_fn():
            def _collate(examples):
                return self.tokenizer.pad(examples, return_tensors="pt")

            return _collate

        dataset = datasets.load_dataset(
            pkg_resources.resource_filename("bert-squeeze", f"data/{self.dataset_config.name}_dataset.py"),
            self.dataset_config.split
        )
        tokenized_dataset = dataset.map(
            lambda x: self.tokenizer(x[self.dataset_config.text_col], padding="max_length", max_length=self.max_length,
                                     truncation=True)
        )
        tokenized_dataset.set_format(type='torch', columns=["input_ids", "token_type_ids", "attention_mask", "id"])
        return DataLoader(tokenized_dataset["train"], collate_fn=_collate_fn(), batch_size=self.batch_size,
                          drop_last=True, num_workers=0)

    def label_dataset(self):
        """
        Annotating the unlabeled dataset using the fine tuned teacher. We return a dictionary
        mapping the document id to its corresponding label.
        """
        output_data = {}
        # TODO: filter out samples with too low probability
        train_loader = self.get_dataloader()
        for batch in tqdm(train_loader, total=len(train_loader), desc="Labeling samples"):
            with torch.no_grad():
                logits = self.model(**batch.to(self.device))
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(axis=-1)

                ids = batch["id"]
                for idx, label in zip(ids.tolist(), preds.tolist()):
                    output_data[idx] = label

                return output_data

        return output_data

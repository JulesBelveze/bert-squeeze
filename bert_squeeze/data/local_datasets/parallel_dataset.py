import json
from typing import List

import datasets

_DESCRIPTION = "Dataset containing parallel data."


class ParallelConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for the below dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ParallelConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class ParallelDataset(datasets.GeneratorBasedBuilder):
    """
    Dataset for parallel distillation
    """

    BUILDER_CONFIG_CLASS = ParallelConfig
    BUILDER_CONFIGS = [
        ParallelConfig(name="default", description=_DESCRIPTION, data_dir="parallel/"),
        ParallelConfig(
            name="debug",
            description="small chunk of the 'default' configuration.",
            data_dir="debug",
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "translation": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": self.config.data_dir + "train.json",
            "test": self.config.data_dir + "test.json",
            "validation": self.config.data_dir + "validation.json",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, "r") as reader:
            data = json.load(reader)

        for id, row in enumerate(data):
            yield id, {
                "text": row["text"],
                "translation": row["translation"],
                "lang": row["lang"],
            }

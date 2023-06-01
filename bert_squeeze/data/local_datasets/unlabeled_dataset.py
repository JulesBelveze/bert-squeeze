from typing import List

import datasets
import pandas as pd

_DESCRIPTION = "Helper dataset to perform soft distillation."


class UnlabeledConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for the below dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(UnlabeledConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class DatasetUnlabeled(datasets.GeneratorBasedBuilder):
    """
    Dataset to use for soft distillation.
    """

    BUILDER_CONFIG_CLASS = UnlabeledConfig
    BUILDER_CONFIGS = [
        UnlabeledConfig(name="default", description=_DESCRIPTION, data_dir="unlabeled/"),
        UnlabeledConfig(
            name="debug",
            description="small chunk of the 'default' configuration.",
            data_dir="debug/",
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"text": datasets.Value("string"), "id": datasets.Value("int16")}
            ),
            supervised_keys=None,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls_to_download = {"train": self.config.data_dir + "train.csv"}
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        df = pd.read_csv(filepath)

        for id, row in df.iterrows():
            yield id, {"text": row["text"], "id": id}

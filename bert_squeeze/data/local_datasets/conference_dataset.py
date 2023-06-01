from typing import List

import datasets
import pandas as pd

_DESCRIPTION = (
    "Dataset used for testing purposes. Taken from here: "
    "https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/title_conference.csv"
)


class ConferenceConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for the conference dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ConferenceConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class ConferenceDataset(datasets.GeneratorBasedBuilder):
    """
    Conference dataset
    """

    BUILDER_CONFIG_CLASS = ConferenceConfig
    BUILDER_CONFIGS = [
        ConferenceConfig(
            name="default", description=_DESCRIPTION, data_dir="classification/"
        ),
        ConferenceConfig(
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
                {
                    "title": datasets.Value("string"),
                    "label": datasets.ClassLabel(
                        names=['ISCAS', 'INFOCOM', 'WWW', 'SIGGRAPH', 'VLDB']
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": self.config.data_dir + "train.csv",
            "test": self.config.data_dir + "test.csv",
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
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        df = pd.read_csv(filepath)

        for id, row in df.iterrows():
            yield id, {"title": row["Title"], "label": row["Conference"]}

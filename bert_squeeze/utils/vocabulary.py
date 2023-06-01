import logging
from typing import List, Optional


class Vocabulary(object):
    """
    Vocabulary class

    Args:
        path_to_voc (str):
            path to the vocabulary file to load
        max_words (int):
            maximum number of words to use to build the vocabulary
    """

    def __init__(
        self,
        path_to_voc: Optional[str] = None,
        max_words: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.max_words = max_words if max_words is not None else 10e7
        self.vocabulary = {"UNK": 0}

        if path_to_voc is not None:
            self.load_vocabulary(path_to_voc)

    def build_vocabulary(self, corpus: List[List[str]]) -> None:
        """
        Method that builds a vocabulary from a corpus of texts.

        Args:
            corpus (List[List[text]]):
                List of tokenized documents from the corpus.
        """
        tokens = set()
        for doc in corpus:
            doc_tokens = set([token for token in doc])
            tokens |= doc_tokens

        for elt in tokens:
            self.add_word(elt)
        logging.info(
            f"Vocabulary successfully built, number of words: {len(self.vocabulary)}"
        )

    def add_word(self, word: str) -> None:
        """
        Method that add a word to the vocabulary.

        Args:
            word (str):
                word to add to the vocabulary
        """
        vocab_len = len(self.vocabulary)
        assert word not in self.vocabulary, f"'{word}' is already in vocabulary."
        if len(self.vocabulary) >= self.max_words:
            logging.warning(
                f"Maximum vocabulary size reached. Not adding '{word}' to vocabulary."
            )
            return

        self.vocabulary[word] = vocab_len

    def load_vocabulary(self, path_to_voc: str) -> None:
        """
        Method that read the vocabulary file and stores the words it contains

        Args:
            path_to_voc (str):
                path to the vocabulary file to load
        """
        with open(path_to_voc, "r") as reader:
            for word in reader:
                if len(self.vocabulary) >= self.max_words:
                    logging.warning("Maximum vocabulary size reached.")
                    return
                self.add_word(word)
        logging.info(f"Vocabulary successfully loaded from '{path_to_voc}'")

    def __getitem__(self, item):
        """"""
        return self.vocabulary.get(item, 0)

import logging
from typing import Iterable

from tqdm import tqdm


class Vocabulary(object):
    def __init__(self, path_to_voc: str = None, max_words: int = None, *args, **kwargs):
        self.max_words = max_words if max_words is not None else 10e7
        self.vocabulary = {"UNK": 0}

        if path_to_voc is not None:
            self.load_vocabulary(path_to_voc)

    def build_vocabulary(self, corpus: Iterable) -> None:
        """"""
        tokens = set()
        for doc in corpus:
            doc_tokens = set([token.text for token in doc])
            tokens |= doc_tokens

        for elt in tokens:
            self.add_word(elt)
        logging.info(f"Vocabulary successfully built, number of words: {len(self.vocabulary)}")

    def add_word(self, word: str) -> None:
        """"""
        vocab_len = len(self.vocabulary)
        assert word not in self.vocabulary, f"'{word}' is already in vocabulary."
        if len(self.vocabulary) >= self.max_words:
            logging.warning(f"Maximum vocabulary size reached. Not adding '{word}' to vocabulary.")
            return

        self.vocabulary[word] = vocab_len

    def load_vocabulary(self, path_to_voc: str) -> None:
        """"""
        with open(path_to_voc, "r") as reader:
            for word in reader:
                if len(self.vocabulary) >= self.max_words:
                    logging.warning(f"Maximum vocabulary size reached.")
                    return
                self.add_word(word)
        logging.info(f"Vocabulary successfully loaded from '{path_to_voc}'")

    def __getitem__(self, item):
        """"""
        return self.vocabulary.get(item, 0)

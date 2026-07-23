import torch
from omegaconf import OmegaConf
from transformers import BertConfig

from bert_squeeze.models.custom_transformers.berxit import BerxitModel
from bert_squeeze.models.custom_transformers.deebert import DeeBertModel
from bert_squeeze.models.custom_transformers.theseus_bert import TheseusBertModel
from bert_squeeze.models.lt_berxit import LtBerxit
from bert_squeeze.models.lt_deebert import LtDeeBert
from bert_squeeze.models.lt_theseus_bert import LtTheseusBert


def _bert_config(num_hidden_layers: int = 1) -> BertConfig:
    return BertConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=num_hidden_layers,
        num_labels=2,
        vocab_size=32,
    )


def _training_config(**overrides):
    config = {
        "logging_steps": 2,
        "accumulation_steps": 1,
        "objective": "ce",
        "lr_scheduler": False,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "token_type_ids": torch.zeros((2, 4), dtype=torch.long),
    }


def test_deebert_loads_and_uses_a_custom_pretrained_encoder(tmp_path):
    source_encoder = DeeBertModel(_bert_config())
    source_encoder.save_pretrained(tmp_path)

    module = LtDeeBert(
        training_config=_training_config(train_highway=False, early_exit_entropy=-1.0),
        pretrained_model=str(tmp_path),
        num_labels=2,
    )
    logits, _, _ = module(**_inputs())

    assert module.model is module.bert
    assert torch.equal(
        module.bert.embeddings.word_embeddings.weight,
        source_encoder.embeddings.word_embeddings.weight,
    )
    assert logits.shape == (2, 2)


def test_berxit_loads_and_uses_a_custom_pretrained_encoder(tmp_path):
    source_encoder = BerxitModel(_bert_config())
    source_encoder.save_pretrained(tmp_path)

    module = LtBerxit(
        training_config=_training_config(train_highway=False, early_exit_entropy=-1.0),
        pretrained_model=str(tmp_path),
        num_labels=2,
    )
    logits, _, _, _ = module(**_inputs())

    assert module.model is module.bert
    assert torch.equal(
        module.bert.embeddings.word_embeddings.weight,
        source_encoder.embeddings.word_embeddings.weight,
    )
    assert logits.shape == (2, 2)


def test_theseus_loads_and_uses_a_custom_pretrained_encoder(tmp_path):
    source_encoder = TheseusBertModel(_bert_config(num_hidden_layers=6))
    source_encoder.save_pretrained(tmp_path)

    module = LtTheseusBert(
        training_config=_training_config(),
        pretrained_model=str(tmp_path),
        num_labels=2,
        replacement_scheduler=OmegaConf.create(
            {"type": "constant", "replacing_rate": 1.0}
        ),
    )
    logits = module(**_inputs())

    assert module.model is module.encoder
    assert torch.equal(
        module.encoder.embeddings.word_embeddings.weight,
        source_encoder.embeddings.word_embeddings.weight,
    )
    assert logits.shape == (2, 2)

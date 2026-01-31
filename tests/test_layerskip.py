from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from bert_squeeze.models.lt_layerskip import LtLayerSkip
from bert_squeeze.utils.callbacks.layerskip_curriculum import LayerSkipCurriculumCallback


def _training_config():
    return OmegaConf.create(
        {
            "logging_steps": 1,
            "accumulation_steps": 0,
            "discriminative_learning": False,
            "weight_decay": 0.0,
            "learning_rates": [1e-4],
            "adam_eps": 1e-8,
            "lr_scheduler": False,
            "warmup_ratio": 0.0,
            "layer_lr_decay": 0.95,
        }
    )


def test_dropout_schedule():
    model = LtLayerSkip(
        training_config=_training_config(),
        pretrained_model="bert-base-uncased",
        num_labels=2,
        p_max=0.1,
    )
    probs = model._compute_dropout_schedule()

    assert probs[0] == pytest.approx(0.0)
    assert probs[-1] == pytest.approx(0.1, abs=1e-6)
    assert probs[6] < probs[7]


def test_forward_all_layers():
    model = LtLayerSkip(
        training_config=_training_config(),
        pretrained_model="bert-base-uncased",
        num_labels=2,
    )
    model.train()

    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids, attention_mask)
    assert isinstance(outputs, tuple)
    assert len(outputs) == model.num_layers
    assert outputs[0].shape == (2, 8, model.model_config.hidden_size)


def test_loss_scales():
    model = LtLayerSkip(
        training_config=_training_config(),
        pretrained_model="bert-base-uncased",
        num_labels=2,
        e_scale=0.2,
    )
    scales = model._compute_loss_scales()

    assert scales[-1] > scales[0]
    assert scales.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_rotational_curriculum():
    callback = LayerSkipCurriculumCallback(
        curriculum_type="rotational", rotation_period=11
    )
    module = SimpleNamespace(num_layers=12, curriculum_mask=torch.zeros(12))
    trainer = SimpleNamespace(
        global_step=5, max_steps=100, estimated_stepping_batches=100
    )

    callback.on_train_batch_start(trainer, module, None, 0)

    assert module.curriculum_mask[5].item() == pytest.approx(1.0)
    assert module.curriculum_mask[11].item() == pytest.approx(1.0)


def test_early_exit_inference():
    model = LtLayerSkip(
        training_config=_training_config(),
        pretrained_model="bert-base-uncased",
        num_labels=2,
        exit_layer=6,
        inference_mode=True,
    )
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids, attention_mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 8, model.model_config.hidden_size)

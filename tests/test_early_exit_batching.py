import pytest
import torch
from transformers import AutoConfig

from bert_squeeze.models.custom_transformers.berxit import BerxitModel
from bert_squeeze.models.custom_transformers.deebert import DeeBertModel
from bert_squeeze.utils.losses.losses import entropy


def _tiny_bert_config():
    return AutoConfig.for_model(
        "bert",
        num_labels=3,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=100,
    )


@pytest.mark.parametrize("model_cls", [DeeBertModel, BerxitModel])
def test_batched_early_exit_matches_single_sample_inference(model_cls):
    """A sample must get its own ramp output, whichever neighbours exit with it."""
    torch.manual_seed(0)
    model = model_cls(_tiny_bert_config(), inference=True).eval()
    input_ids = torch.randint(1, 100, (8, 6))

    with torch.no_grad():
        # An infinite threshold makes every sample exit at the first ramp, which
        # exposes the first-layer entropies used to pick a splitting threshold.
        model.encoder.set_early_exit_entropy(float("inf"))
        first_layer = model(input_ids=input_ids).ramps_exits
        entropies = torch.stack([ramp.entropy.reshape(()) for ramp in first_layer])
        threshold = entropies.median().item()
        exits_first = entropies < threshold
        # The exiting samples must not be the leading rows, otherwise positional
        # indexing happens to be right and the test proves nothing.
        assert exits_first.any() and not exits_first[: int(exits_first.sum())].all()

        model.encoder.set_early_exit_entropy(threshold)
        batched = model(input_ids=input_ids).ramps_exits
        singles = [model(input_ids=input_ids[i : i + 1]).ramps_exits[0] for i in range(8)]

    for batch_ramp, single_ramp in zip(batched, singles):
        assert torch.allclose(
            batch_ramp.logits.reshape(-1), single_ramp.logits.reshape(-1), atol=1e-5
        )


def test_entropy_does_not_depend_on_the_sign_of_the_logits():
    """All-positive logits used to be read as unnormalized probabilities."""
    positive = torch.tensor([[0.0154, 0.0008, 0.0180]])
    mixed = torch.tensor([[0.0154, 0.0008, 0.0180], [0.0162, -0.0107, -0.0127]])
    assert torch.allclose(entropy(positive)[0], entropy(mixed)[0], atol=1e-6)

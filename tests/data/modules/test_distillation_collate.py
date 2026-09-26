import torch

from bert_squeeze.data.modules.distillation_module import DistillationDataModule


def test_pad_sequences_handles_scalar_and_sequence_columns():
    """Scalar columns (labels) must not be treated as sequences to pad."""
    batch = [
        {"t_input_ids": [1, 2, 3], "t_labels": 0, "s_input_ids": [4, 5]},
        {"t_input_ids": [6], "t_labels": 2, "s_input_ids": [7, 8, 9]},
    ]

    padded = DistillationDataModule.pad_sequences(batch, padding_value=-100)

    # scalar labels are stacked, not padded
    assert padded["t_labels"].tolist() == [0, 2]
    # sequences are right-padded to the batch max length
    assert padded["t_input_ids"].tolist() == [[1, 2, 3], [6, -100, -100]]
    assert padded["s_input_ids"].tolist() == [[4, 5, -100], [7, 8, 9]]
    assert padded["t_input_ids"].dtype == torch.long

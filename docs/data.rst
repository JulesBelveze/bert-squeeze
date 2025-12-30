Data
======================

Multi-source loading
--------------------

`Seq2SeqTransformerDataModule` now accepts multiple dataset sources through
`dataset_config.data_path`. Provide a list of paths plus `data_format`
(`disk`, `hub`, `json`, `csv`, etc.) and set `combine_strategy` to either
`concatenate` (default) or `interleave`. When omitted, legacy single-path behaviour
still uses `dataset_config.path` with `datasets.load_dataset`.

Example Hydra snippet::

    data:
      dataset_config:
        data_path:
          - data/english_disk
          - data/french_disk
        data_format: disk
        combine_strategy: interleave

All datasets must expose the same splits (train/validation/test). Use `data_format`
as a list when each path uses a different storage format.

bert_squeeze.data.modules
----------------------------


.. automodule:: bert_squeeze.data.modules.distillation_module
    :members:
    :exclude-members: test_dataloader, train_dataloader, val_dataloader

.. automodule:: bert_squeeze.data.modules.lr_module
    :members:
    :exclude-members: test_dataloader, train_dataloader, val_dataloader

.. automodule:: bert_squeeze.data.modules.lstm_module
    :members:
    :exclude-members: test_dataloader, train_dataloader, val_dataloader, collate_fn

.. automodule:: bert_squeeze.data.modules.transformer_module
    :members:
    :exclude-members: test_dataloader, train_dataloader, val_dataloader

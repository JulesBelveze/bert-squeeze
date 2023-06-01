Models
======================

We provide the users numerous models implementation, ranging from a very simple logistic regression to a more complex
Transformer-based model with exiting strategies. Most of the models are either direct papers implementation or code
taken from a repository and rewritten into a more concise way.

If you feel like a model is missing and would be a great addition to the repository, please open an issue!

Lightning Models
----------------------------

The following models are `LightningModule` wrapping up PyTorch models and defining the corresponding training logic.

.. autoclass:: bert_squeeze.models.base_lt_module.BaseTransformerModule

.. autoclass:: bert_squeeze.models.lr.BowLogisticRegression

.. autoclass:: bert_squeeze.models.lstm.LtLSTM

.. autoclass:: bert_squeeze.models.lt_bert.LtCustomBert

.. autoclass:: bert_squeeze.models.lt_deebert.LtDeeBert

.. autoclass:: bert_squeeze.models.lt_distilbert.LtCustomDistilBert

.. autoclass:: bert_squeeze.models.lt_fastbert.LtFastBert

.. autoclass:: bert_squeeze.models.lt_theseus_bert.LtTheseusBert

Custom Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following models are PyTorch implementations of some specific Transformer-based models.

.. automodule:: bert_squeeze.models.custom_transformers.bert
   :members:

.. automodule:: bert_squeeze.models.custom_transformers.deebert
   :members:

.. automodule:: bert_squeeze.models.custom_transformers.fastbert
   :members:

.. automodule:: bert_squeeze.models.custom_transformers.theseus_bert
   :members:

Model Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following sub-package contains implementation of multiple layers that can be reused across different model
architectures.

.. automodule:: bert_squeeze.models.layers.classifier
   :members:

.. automodule:: bert_squeeze.models.layers.mha
   :members:


Distillation
======================

We provide the user with multiple classes to distil a teacher model into a student one. There are some already
implemented distillation logic. However, if you want to use a custom technique you simply need to extend the
`BaseDistiller` class.


bert_squeeze.distillation
----------------------------

.. autoclass:: bert_squeeze.distillation.distiller.BaseDistiller

.. autoclass:: bert_squeeze.distillation.distiller.Distiller

.. autoclass:: bert_squeeze.distillation.distiller.ParallelDistiller

.. toctree:: distillation.utils
    :maxdepth: 1
    :caption: Distillation Utilities

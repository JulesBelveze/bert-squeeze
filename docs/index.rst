*bertsqueeze*
========================================

.. image:: https://img.shields.io/github/stars/JulesBelveze/bert-squeeze.svg?style=social&label=Star&maxAge=2500
   :target: https://github.com/JulesBelveze/bert-squeeze

Project
------------

**Bert-squeeze** is a repository aiming to provide code to reduce the size of Transformer-based models or decrease their
latency at inference time.

It gathers a non-exhaustive list of techniques such as `distillation <https://en.wikipedia.org/wiki/Knowledge_distillation/>`_
, `pruning <https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)/>`_,
`quantization <https://en.wikipedia.org/wiki/Quantization/>`_, early-exiting, ... The repo is built using
`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ and `Transformers <https://huggingface.co/transformers/>`_.

Content
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   tutorials/Training
   tutorials/Distillation


.. toctree::
   :maxdepth: 1
   :caption: Bertsqueeze

   overview
   concepts
   assistants
   distillation
   models

.. toctree::
   :maxdepth: 2
   :caption: API References

   api
   GitHub Repository <https://github.com/JulesBelveze/bert-squeeze>



Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`

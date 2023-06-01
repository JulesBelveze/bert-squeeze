Concepts
===========

Transformers
--------------

If you have never heard of Transformers then I can only recommend you to read this amazing
`blog post <https://jalammar.github.io/illustrated-transformer/>`_ which will give you the keys to understand the main
concepts behind those models. If you later on want to dig deeper there is this awesome lecture was given by
Stanford available `here <https://www.youtube.com/watch?v=ptuGllU5SQQ/>`_.

Distillation
--------------

The idea of distillation is to train a small network to mimic a big network by trying to replicate its outputs. The
repository provides the ability to transfer knowledge from any model to any other (if you need a model that is not
within the models folder just write your own).

The repository also provides the possibility to perform *soft distillation* or *hard distillation* on an unlabeled
dataset. In the soft case, we use the probabilities of the teacher as a target. In the hard one, we assume that the
teacher's predictions are the actual label.

You can find these implementations under the :file:`distillation/` folder.

Quantization
--------------

Neural network quantization is the process of reducing the weights precision in the neural network. The repo has two
callbacks one for dynamic quantization and one for quantization-aware training.

You can find those implementations under the :file:`utils/callbacks/` folder.

Pruning
--------------

Pruning neural networks consist of removing weights from trained models to compress them. This repo features various
pruning implementations and methods such as head-pruning, layer dropping, and weights dropping.

You can find those implementations under the :file:`utils/callbacks/` folder.
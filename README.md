<img src="./images/bert-squeeze.png" height="25%" align="right"/>

<p align="center">
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" height="20px">
   <img src="https://img.shields.io/badge/PyTorch Lightning-792EE5?style=for-the-badge&logo=PyTorch Lightning&logoColor=white" height=20px>
</p>

# Bert-squeeze

**Bert-squeeze** is a repository aiming to provide code to reduce the size of Transformer-based models or decrease their
latency at inference time.

It gathers a non-exhaustive list of techniques such as distillation, pruning, quantization, early-exiting. The repo is
written using [PyTorch Lightning](https://www.pytorchlightning.ai)
and [Transformers](https://huggingface.co/transformers/).

# About the project

As a heavy user of transformer-based models (which are truly amazing from my point of view) I always struggled to put
those heavy models in production while having a decent inference speed. There are of course a bunch of existing
libraries to optimize and compress transformer-based models ([ONNX](https://github.com/onnx/onnx)
, [distiller](https://github.com/IntelLabs/distiller), [compressors](https://github.com/elephantmipt/compressors)
, [KD_Lib](https://github.com/SforAiDl/KD_Lib), ... ). \
I started this project because of the need to reduce the latency of models integrating transformers as subcomponents.
For this reason, this project aims at providing implementations to train various transformer-based models (and others)
using PyTorch Lightning but also to distill, prune, and quantize models. \
I chose to write this repo with Lightning because of its growing trend, its flexibility, and the very few repositories
using it. It currently only handles sequence classification models, but support for other tasks and custom architectures
is [planned](https://github.com/JulesBelveze/bert-squeeze/projects/10).

# Installation

First download the repository:

```commandline
git clone https://github.com/JulesBelveze/bert-squeeze.git
```

and then install dependencies using [poetry](https://python-poetry.org/docs/):

```commandline
poetry install
```

You are all set!

# Quickstarts

You can find a bunch of already prepared configurations under the examples folder. Just choose the one you need and run
the following:

```commandline
python3 -m bert-squeeze.main -cp=examples -cn=wanted_config
```

Disclaimer: I have not extensively tested all procedures and thus do not guarantee the performance of every implemented
method.

# Concepts

### Transformers

If you never heard of it then I can only recommend you to read this
amazing [blog post](https://jalammar.github.io/illustrated-transformer) and if you want to dig deeper there is this
awesome lecture was given by Stanford available [here](https://www.youtube.com/watch?v=ptuGllU5SQQ).

### Distillation

The idea of distillation is to train a small network to mimic a big network by trying to replicate its outputs. The
repository provides the ability to transfer knowledge from any model to any other (if you need a model that is not
within the `models` folder just write your own).

The repository also provides the possibility to perform soft-distillation or hard-distillation on an unlabeled dataset.
In the soft case, we use the probabilities of the teacher as a target. In the hard one, we assume that the teacher's
predictions are the actual label.

You can find these implementations under the `distillation/` folder.

### Quantization

Neural network quantization is the process of reducing the weights precision in the neural network. The repo has two
callbacks one for dynamic quantization and one for quantization-aware training (using
the [Lightning callback](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.QuantizationAwareTraining.html#pytorch_lightning.callbacks.QuantizationAwareTraining))
.

You can find those implementations under the `utils/callbacks/` folder.

### Pruning

Pruning neural networks consist of removing weights from trained models to compress them. This repo features various
pruning implementations and methods such as head-pruning, layer dropping, and weights dropping.

You can find those implementations under the `utils/callbacks/` folder.

# Contributions and questions

If you are missing a feature that could be relevant to this repo, or a bug that you noticed feel free to open a PR or
open an issue. As you can see in the [roadmap](https://github.com/JulesBelveze/bert-squeeze/projects/1) there are a
bunch more features to come :smiley:

Also, if you have any questions or suggestions feel free to ask!

# References

1. Alammar, J (2018). _The Illustrated Transformer_ [Blog post]. Retrieved
   from https://jalammar.github.io/illustrated-transformer/
2. stanfordonline (2021) _Stanford CS224N NLP with Deep Learning | Winter 2021 | Lecture 9 - Self- Attention and
   Transformers_. [online video] Available at: https://www.youtube.com/watch?v=ptuGllU5SQQ
3. Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric
   Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Jamie Brew (2019). [_HuggingFace's Transformers:
   State-of-the-art Natural Language Processing_](http://arxiv.org/abs/1910.03771)
4. Hassan Sajjad and Fahim Dalvi and Nadir Durrani and Preslav Nakov (2020). [_Poor Man's BERT Smaller and Faster
   Transformer Models_](https://arxiv.org/abs/2004.03844)
5. Angela Fan and Edouard Grave and Armand Joulin (2019). [_Reducing Transformer Depth on Demand with Structured
   Dropout_](http://arxiv.org/abs/1909.11556)
6. Paul Michel and Omer Levy and Graham Neubig (2019). [_Are Sixteen Heads Really Better than
   One?_](http://arxiv.org/abs/1905.10650)
7. Fangxiaoyu Feng and Yinfei Yang and Daniel Cer and Naveen Arivazhagan and Wei Wang (2020). [_Language-agnostic BERT
   Sentence Embedding_](https://arxiv.org/abs/2007.01852)
8. Weijie Liu and Peng Zhou and Zhe Zhao and Zhiruo Wang and Haotang Deng and Qi Ju (2020). [_FastBERT: a
   Self-distilling {BERT} with Adaptive Inference Time_](https://arxiv.org/abs/2004.02178). \
   Repository: https://github.com/BitVoyage/FastBERT
9. Xu, Canwen and Zhou, Wangchunshu and Ge, Tao and Wei, Furu and Zhou, Ming (2020). [_{BERT}-of-Theseus: Compressing
   {BERT} by Progressive Module Replacing_](https://www.aclweb.org/anthology/2020.emnlp-main.633)

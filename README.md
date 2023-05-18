# LeTI: Learning to Generate from Textual Interactions

Official repo for paper [LeTI: Learning to Generate from Textual Interactions](https://arxiv.org/abs/2305.10314).

This repo contains code that can be used to reproduce the experiments in the paper. Train/evaluation code are written using Jax/Flax to train on Google Cloud's TPU VM instances.
Research is supported with Cloud TPUs from [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

**WARNING**
Training and evaluation of LeTI requires executing untrusted model-generated code. Users are strongly encouraged to sandbox the code execution so that it does not perform destructive actions on their host or network.

## Setup

You can setup your Google Cloud TPU and Storage following [docs/SETUP.md](docs/SETUP.md). Alternatively, you may also adapt the released code to your specific computing setup.

## Dataset

You can prepare datasets for training and evaluation following instructions in [docs/DATA.md](docs/DATA.md).

## Model

Since the training and evaluation code is implemented using Jax/Flax, you will need to convert huggingface model checkpoints (pytorch) into [T5X](https://github.com/google-research/t5x) format, following instructions in [docs/MODEL.md](docs/MODEL.md).

## Training and Evaluation

You can follow [docs/TRAIN.md](docs/TRAIN.md) and [docs/EVAL.md](docs/EVAL.md) to train or evaluate a specific model.


## Citation

```
@article{Wang2023LeTI,
  title={LeTI: Learning to Generate from Textual Interactions},
  author={Xingyao Wang and Hao Peng and Reyhaneh Jabbarvand and Heng Ji},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.10314},
}
```

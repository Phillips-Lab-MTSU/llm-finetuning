# LLM-Finetuning
A brief tutorial for fine-tuning LLMs on small cluster GPU hardware.

## Overview

These are a set of instructions for downloading/training LLMs on our MTSU cluster resources. Most of the material is adapted from the [NeurIPS 2023 LLM Efficiency Challenge](https://github.com/ayulockin/neurips-llm-efficiency-challenge). The tools have changed in several ways from Aug 2023 to November 2023. I've tried to make a set of instructions consistent with the newest version of `lit-gpt` that we can possibly support at the moment (one that still runs on PyTorch 2.1 instead of 2.2 since 2.2 is still not `stable`).

Overall, `lit-gpt` is nice set of relatively simple scripts which can be hacked to meet your specific needs once you have used them as part of the tutorials below. Here is a link to the project [repo](https://github.com/Lightning-AI/lit-gpt/tree/6178c7cc58ba82e5cce138e7a3159c384e2d3b0f) for the specific version we will be using.

Instructions are broken into a two categories:

* [Scripting.md](Scripting.md)

    This tutorial covers how to use scripting scaffolded by `lit-gpt` to download/convert model checkpoints, prepare common data sets, finetune using single- or multi-GPU training, and evaluate using some common benchmarks. The point is to demonstrate a scalable method. First, how to train `Eleuther/pythia-160m` on a single 2080 Ti (12Gb) on hamilton, and then also how to train `Eleuther/pythia-410m` across two 2080 Ti (12Gb) nodes. The latter model does not fit on a single 2080 Ti and so necessitates distributed training.
  
* [JupyterLab.md](JupyterLab.md)
  
  This tutorial covers how to run a single-GPU interactive JupyterLab session in which you can use `lit-gpt` to download/convert model checkpoints, prepare common data sets, finetune using single- or multi-GPU training, and evaluate using some common benchmarks. Note that this approachs is inherently limited by the single-GPU bottleneck. Some small models like `Eleuther/pythia-160m` will train on a single 2080 Ti (12Gb) on bababage/hamilton, `Eleuther/pythia-1b` will train on a single A5000 (24Gb) on babbage, but larger models will need the A100 (80Gb) on hamilton.

All repository components Copyright &copy; 2023 Phillips Lab @ MTSU

See [LICENSE](LICENSE) for attribution details.
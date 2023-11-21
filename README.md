# llm-finetuning
A tutorial for fine-tuning LLMs on small cluster GPU hardware

# Overview

These are a set of instructions for downloading/training LLMs on our MTSU cluster resources. Most of the material is adapted from the [NeurIPS 2023 LLM Efficiency Challenge](https://github.com/ayulockin/neurips-llm-efficiency-challenge).

# Running on Babbage/Hamilton

You will need to have a working stack with PyTorch >= 2.1 since we will be utilizing [`lit-gpt`](https://github.com/Lightning-AI/lit-gpt) for our training framework which requires this minimal PyTorch version.

Custom stack is available on Babbage/Hamilton and compatible with [`apptainer`](https://apptainer.org/):
`/home/shared/sif/csci-2023-10-20.sif`

I recommend logging into our JupyterHub using [azuread](https://jupyterhub.cs.mtsu.edu/azuread/) then log into Babbage using a terminal, and start a JupyterLab session using the stack above. I utilize the following command to start my job to get an A5000 on Babbage:

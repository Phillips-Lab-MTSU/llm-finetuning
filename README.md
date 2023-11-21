# LLM-Finetuning
A brief tutorial for fine-tuning LLMs on (not-so) small (yet!) cluster GPU hardware.

## Overview

These are a set of instructions for downloading/training LLMs on our MTSU cluster resources. Most of the material is adapted from the [NeurIPS 2023 LLM Efficiency Challenge](https://github.com/ayulockin/neurips-llm-efficiency-challenge). The tools have changed in several ways from Aug 2023 to November 2023. I've tried to make a set of instructions consistent with the newest version of `lit-gpt` that we can use (one that still runs on PyTorch 2.1 instead of 2.2 since 2.2 is still not `stable`). 

## Running on Babbage/Hamilton

You will need to have a working stack with PyTorch >= 2.1 since we will be utilizing [`lit-gpt`](https://github.com/Lightning-AI/lit-gpt) for our training framework which requires this minimal PyTorch version.

I've built a custom stack which is available on Babbage/Hamilton and compatible with [`apptainer`](https://apptainer.org/) and is located in a shared directory:
`/home/shared/sif/csci-2023-10-20.sif`

The above stack is quite large since we utiilize it for so many purposes at MTSU, but if you wanted to build the image using Docker, the [stack repo for this specific version is here](https://github.com/Phillips-Lab-MTSU/CSCI-MTSU-JupyterHub/tree/24a76ca55d4c289359c2a84cf6eaf8b0140e21cf). Once you build the image using Docker, you can use apptainer to convert it to a `sif` and upload it to wherever you would like. However, a publicly downloadable version is provided [here in the cloud](https://data.phillips-lab.org/sif/csci-2023-10-20.sif).

I recommend logging into our JupyterHub using [azuread](https://jupyterhub.cs.mtsu.edu/azuread/) then log into Babbage using a terminal and start a JupyterLab session using the stack above. I utilize `tmux` and the following command to start my job to get an A5000 on Babbage:
```
srun \
    -G 1 \
    -p research-gpu \
    -t 2400 \
    -x c[11-14] \
    apptainer run \
        --nv \
        --env NB_UID=${UID} \
        --env NB_USER=${USER} \
        --env HF_HOME=${HOME}/hf_home \
        --bind /home/shared/checkpoints:/home/checkpoints \
        --writable-tmpfs \
        -H jlab:${HOME} \
        --env XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/pkgs/cuda-toolkit" \
        --env NOTEBOOK_ARGS="--NotebookApp.base_url=/azuread/user/${USER}/proxy/absolute/9000/ --NotebookApp.custom_display_url=https://jupyterhub.cs.mtsu.edu" \
        /home/shared/sif/csci-2023-10-20.sif
```
or if trying to get the A100 on Hamilton (currently required for 7B+ parameter models):
```
srun \
    -G 1 \
    -p a100 \
    -t 2400 \
    apptainer run \
        --nv \
        --env NB_UID=${UID} \
        --env NB_USER=${USER} \
        --env HF_HOME=${HOME}/hf_home \
        --bind /home/shared/checkpoints:/home/checkpoints \
        --writable-tmpfs \
        -H jlab:${HOME} \
        --env XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/pkgs/cuda-toolkit" \
        --env NOTEBOOK_ARGS="--NotebookApp.base_url=/azuread/user/${USER}/proxy/absolute/9000/ --NotebookApp.custom_display_url=https://jupyterhub.cs.mtsu.edu" \
        /home/shared/sif/csci-2023-10-20.sif
```

Once the job is in the queue and started up, I check which node I have been allocated using `squeue` and then log out and back in, adding the appropriate forwarding argument for that node (eg. `-L 9000:c17:8888` if I was allocated node `c17`). Note also that I have a subdirectory named `jlab` which I use for running the session: this just ensures I don't clobber any other files in my home directory unintentionally. You can change to this some other custom directory of your choice.

## Installation of Packages

You will need to customize the environment a little:
```
pip install --user jsonargparse[signatures] sentencepiece bitsandbytes==0.41.0
MAX_JOBS=8 pip install --user flash-attn --no-build-isolation
```
Prep `lit-gpt` by pulling a version compatible with flash-attn 2. Note that the newest version of `lit-gpt` dropped the flash-attn dependency, but then requires PyTorch 2.2. I will update this step when 2.2 rolls from nightly into stable, but I don't feel comfortable putting students/faculty on a non-stable platform. For now, use this for documentation then: [browse repo at commit below](https://github.com/Lightning-AI/lit-gpt/tree/6178c7cc58ba82e5cce138e7a3159c384e2d3b0f)
```
git clone https://github.com/Lightning-AI/lit-gpt.git
cd lit-gpt
git checkout 6178c7cc58ba82e5cce138e7a3159c384e2d3b0f
cd ..
```

## Download/Prep Model

Note that I have downloaded some models already in `/home/shared/checkpoints`, so check there first before running the next command:
```
python lit-gpt/scripts/download.py \
    --repo_id tiiuae/falcon-7b
```
You can use the pre-downloaded version instead here (or modify to point to your downloaded version) as the following command loads the model into CPU RAM:
```
python lit-gpt/scripts/convert_hf_checkpoint.py \
    --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b
```
Check that the model works (will utilize GPU if available - about 15G for falcon so inference, but not training, on A5000 is possible):
```
python lit-gpt/generate/base.py \
    --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b \
    --prompt "Tell me an interesting fun fact about earth:"
```

## Download/Prep Data
A small data set (won't be very good, but can be good to quickly check the pipeline).
```
python lit-gpt/scripts/prepare_dolly.py \
    --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b
```

## Fine-tune the LLM
Currently only runs on the A100: ~65G GPU RAM. There will be an issue at this stage where the `generate_prompt()` function cannot be imported due to version skew and interactions with the `lm-eval` package. You might get away with it for now, but once you install the `lm-eval` package for the evaluation step below, it will show up again. Best squash the bug now. You can open `lit-gpt/finetune/lora.py` and comment out the offending line (`from scripts.prepare_alpaca import generate_prompt`), then copy-paste the function from `lit-gpt/scripts/prepare_alpaca.py` directly below the commented line. (I'll work on performing these steps with the latest `lit-gpt` instead soon.)

```
python lit-gpt/finetune/lora.py \
    --data_dir data/dolly
    --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b
    --precision bf16-true
    --out_dir out/lora/falcon-7b
    --quantize "bnb.nf4"
```
Double quantization - a little less memory, and probably not worth it, but this is how it's done.
```
python lit-gpt/finetune/lora.py \
    --data_dir data/dolly \
    --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b \
    --precision bf16-true \
    --out_dir out/lora/falcon-7b \
    --quantize "bnb.nf4-dq"
```

## Model Evaluation
Prep evaluation tools (package `lm-eval`).
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout 115206dc89dad67b8beaa90051fb52db77f0a529
pip install --user --no-cache-dir -e .
cd ..
```

Check the evaluation on the original model.

```
python lit-gpt/eval/lm_eval_harness.py \
    --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" \
    --batch_size 4 \
    --save_filepath results-falcon-7b.json
```

Now for the fine-tuned model. Need to first merge the weights when using lora for finetuning as [explained here](https://github.com/Lightning-AI/lit-gpt/blob/6178c7cc58ba82e5cce138e7a3159c384e2d3b0f/tutorials/finetune_lora.md). Then we can basically proceed in a similar manner as the original model, but be sure to check over the link and see if there is anything specific to your model that needs to be done. (For example, Llama 2 also needs a manual copy step for it's `tokenizer.model` file to complete the merge.)
```
python scripts/merge_lora.py \
  --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b \
  --lora_path out/lora/falcon-7b/lit_model_lora_finetuned.pth" \
  --out_dir out/lora_merged/falcon-7b
```
and the final evaluation step:
```
python lit-gpt/eval/lm_eval_harness.py \
    --checkpoint_dir out/lora_merged/falcon-7b \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" \
    --batch_size 4 \
    --save_filepath results-falcon-7b-finetuned.json
```

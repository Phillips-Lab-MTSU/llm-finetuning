# Scripting
A brief tutorial for scripted finetuning of LLMs on small cluster GPU hardware.

## Overview

This tutorial will walk through the steps in a scripted fashion. 

## Running on Babbage/Hamilton

You will need to have a working stack with PyTorch >= 2.1 since we will be utilizing [`lit-gpt`](https://github.com/Lightning-AI/lit-gpt) for our training framework which requires this minimal PyTorch version.

I've built a custom stack which is available on Babbage/Hamilton and compatible with [`apptainer`](https://apptainer.org/) and is located in a shared directory:
`/home/shared/sif/csci-2023-10-20.sif`

The above stack is quite large since we utiilize it for so many purposes at MTSU, but if you wanted to build the image using Docker, the [stack repo for this specific version is here](https://github.com/Phillips-Lab-MTSU/CSCI-MTSU-JupyterHub/tree/24a76ca55d4c289359c2a84cf6eaf8b0140e21cf). Once you build the image using Docker, you can use apptainer to convert it to a `sif` and upload it to wherever you would like. However, a publicly downloadable version is provided [here in the cloud](https://data.phillips-lab.org/sif/csci-2023-10-20.sif).

I recommend logging into our JupyterHub using [azuread](https://jupyterhub.cs.mtsu.edu/azuread/) then using a **terminal** to log into hamilton (`ssh username@hamilton.cs.mtsu.edu`). You may ssh into babbage as well but you will need to alter the job partition name in the scripts described below from `research-gpu` instead of `research` for that system. You might also want to utilize a `tmux` or `screen` session at this point, but it's optional.

## Clone the Repo

There are a couple of scripts in the repo that we will be using so cloning the repo and then also using the repo subdirectory for your tutorial work will simplify things moving forward.

```
git clone https://github.com/Phillips-Lab-MTSU/llm-finetuning
cd llm-finetuning
```

## Important Script

There is a key script included in the repo which will help us get started: `llm-run.sh`. We will run all of our commands using the `llm-run.sh` script in the next few sections. This script requests one 10-hour job, starts the apptainer stack, and then runs the script/commands which follow as arguments within that containerized environment. Any commands or package installations need to be prefaced by the script in order to make sure you are using the correct software stack during execution.

I highly recommend opening up the `llm-run.sh` script in your favorite text editor and look at the different components. There are **four key components** that would like to point out:

1. At the top of the script, you will see some comments particular to the running jobs using `sbatch` (noninteractively). We will not use `sbatch` at first but these settings will dictate what resources the `srun` command used later in the script can actually request for the current job, and are normally an exact match to those below.

```
# SLURM SUBMIT SCRIPT
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --time=0-10:00:00     # Requested time
#SBATCH --partition=research  # Partition/queue for running the job
#SBATCH --signal=SIGUSR1@90   # Enables auto-requeueing for SLURM/PyTorchLightning
```

2. In the middle you will find some variables for setting up your environment in a way that is helpful when working with `apptainer` images. The `LLM_DATA_DIR` variable is particular to hamilton/babbage and is a shared location where model checkpoints and data sets are stored, and we will mount it inside of our container. The `LLM_CACHE_DIR` is a writable location that we will mount inside of the container to make sure that certain utilities will function (many libraries will attempt to write to `/home/jovyan` rather than your home directory for technical reasons, and this will prevent this issue). Your home directory is automatically mounted when running `apptainer` images, so we don't need to worry about that one. 

```
# data locations
export LLM_DATA_DIR=/home/shared/llm
export LLM_CACHE_DIR=${HOME}/.cache/llm
```

3. Often, we will run jobs *interactively* and therefore not use `sbatch`. The script therefore utilizes `srun` to make this straight-forward, since all *neccessary* arguments for `srun` are in the script, as well as all arguments for the `apptainer` command. These match the settings for `sbatch` above and should generally always be kept in-sync with one-another.

```
srun \
	--ntasks-per-node=1 \
	--partition=research \
	--time=0-10:00:00 \
	apptainer exec \
		--nv \
		--env XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/pkgs/cuda-toolkit" \
		--writable-tmpfs \
		--bind ${LLM_CACHE_DIR}:/home/jovyan \
		--bind ${LLM_DATA_DIR}:${LLM_DATA_DIR} \
		/home/shared/sif/csci-2023-10-20.sif \
		"$@"
```

4. Note the `"$@"` at the end: this means our script is a wrapper which we can use to run our jobs since any commands are passed to `srun/apptainer` for running with the container image. Generally, we can use it to run *interactive* jobs with 1 (2080 Ti) GPU by just prefacing our commands with `salloc --gpus=1 llm-run.sh`, and also running *noninteractive* jobs by prefacing our commands with `sbatch --gpus=1 llm-run.sh`. We can utilize the same script to specify multiple GPUs and accomplish distributed GPU training later on.

## Installation of Packages

You will need to customize the environment a little:
```
salloc --gpus=1 llm-run.sh pip install --user jsonargparse[signatures] sentencepiece bitsandbytes==0.41.0
```

#### Important: `flash-attn` is a package for speeding up transformer architectures, but comes at a cost of requiring A100/H00-level hardware on most of my current testing.

While this might change in the future, **certain 1B models that normally fit on a 24Gb GPU simply will not train with flash-attn installed**. I recommend making a choice based on available hardware: if you have an A100/H100, then you can install it for speed-up, but if you **don't** have an A100/H100, then you should **not** install it to allow for these transformer models to train on smaller-memory hardware. Again, *only run the next command if you have an A100/H100*. 
```
MAX_JOBS=8 salloc --gpus=1 llm-run.sh pip install --user flash-attn --no-build-isolation
```
If you need to reverse the step above, just uninstall that package:
```
salloc --gpus=1 llm-run.sh pip uninstall flash-attn
```

Prep `lit-gpt` by pulling a version compatible with flash-attn 2. Note that the newest version of `lit-gpt` dropped the flash-attn dependency, but then requires PyTorch 2.2. It's important to note that `flash-attn` is used by PyTorch 2.2 if it is installed. I will update this step when 2.2 rolls from nightly into stable, but I don't feel comfortable putting students/faculty on a non-stable platform. For now, use this for documentation then: [browse repo at commit listed below](https://github.com/Lightning-AI/lit-gpt/tree/6178c7cc58ba82e5cce138e7a3159c384e2d3b0f)
```
git clone https://github.com/Lightning-AI/lit-gpt.git
cd lit-gpt
git checkout 6178c7cc58ba82e5cce138e7a3159c384e2d3b0f
cd ..
```

Prep evaluation tools (package `lm-eval`).
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout 115206dc89dad67b8beaa90051fb52db77f0a529
salloc --gpus=1 ../llm-single.sh pip install --user --no-cache-dir -e .
cd ..
```

## Download/Prep a Model

Note that I have downloaded some models already in `/home/shared/llm/checkpoints`, so check there first before running the next commands. In particular, you will have to select which model you would like and use the appropriate checkpoint directory for the model that you download for all subsequent commands. Take a look at the github repo for `lit-gpt` for [options](https://github.com/Lightning-AI/lit-gpt), then continue on...

The following commands will download the model from hugging face and then convert it from hugging face checkpoint format to PyTorch Lightning checkpoint format:
```
salloc --gpus=1 llm-run.sh \
    python lit-gpt/scripts/download.py \
        --repo_id EleutherAI/pythia-160m
salloc --gpus=1 llm-run.sh \
    python lit-gpt/scripts/convert_hf_checkpoint.py \
        --checkpoint_dir checkpoints/EleutherAI/pythia-160m
```

You can use the pre-downloaded version instead (or modify to point to your downloaded version). Note that I will be using the pre-downloaded path from this point forward, so you may need to make alterations (like removing the `/home/shared/llm` prefix from model/data paths). Check that the model works (will utilize the GPU):
```
salloc --gpus=1 llm-run.sh \
    python lit-gpt/generate/base.py \
        --checkpoint_dir /home/shared/llm/checkpoints/EleutherAI/pythia-160m \
        --prompt "Tell me an interesting fun fact about earth:"
```

## Download/Prep a Dataset
A small data set (won't be very good, but can be good to quickly check the training pipeline).
```
salloc --gpus=1 llm-run.sh \
    python lit-gpt/scripts/prepare_dolly.py \
        --checkpoint_dir /home/shared/llm/checkpoints/EleutherAI/pythia-160m \
        --destination_dir /home/shared/llm/data/EleutherAI/pythia-160m/dolly
```

## Finetune the LLM
There will be an issue at this stage where the `generate_prompt()` function cannot be imported due to version skew and interactions with the `lm-eval` package. You might get away with it for now if you remove than package, but once you install the `lm-eval` package for the evaluation step below, it will show up again. Best squash this bug now. Also, I have made a modification to the `lit-gpt/finetune/lora.py` script to allow for parallel training as well which I will refer to later on. The standard script would allow for multiple GPUs on the *same* node by adjusting the `devices` variable, but I added variables and logic to allow the Lightning Farbic module to use multiple GPUs spread across more than one node. To keep this simple, you can just use the command below to patch your script and then everything will work out:
```
patch lit-gpt/finetune/lora.py < lora.patch
```

Here is a **diff** of the changes to `lit-gpt/finetune/lora.py` so that you can see exactly what I am doing for this patch:

```
29c29,44
< from scripts.prepare_alpaca import generate_prompt
---
> #from scripts.prepare_alpaca import generate_prompt
> def generate_prompt(example: dict) -> str:
>     """Generates a standardized message to prompt the model with an instruction, optional input and a
>     'response' field."""
> 
>     if example["input"]:
>         return (
>             "Below is an instruction that describes a task, paired with an input that provides further context. "
>             "Write a response that appropriately completes the request.\n\n"
>             f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
>         )
>     return (
>         "Below is an instruction that describes a task. "
>         "Write a response that appropriately completes the request.\n\n"
>         f"### Instruction:\n{example['instruction']}\n\n### Response:"
>     )
36c51,53
< devices = 1
---
> devices_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
> num_nodes = int(os.environ["SLURM_NNODES"])
> devices = num_nodes * devices_per_node
94c111
<     fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
---
>     fabric = L.Fabric(devices=devices_per_node, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
```

Once you are ready, you can proceed with finetuning. 
Note that this is submitted as a batch job since it will take awhile (it might take up to around 1.5 hours for this particular example).
```
sbatch --gpus=1 llm-run.sh \
    python lit-gpt/finetune/lora.py \
        --data_dir /home/shared/llm/data/EleutherAI/pythia-160m/dolly \
        --checkpoint_dir /home/shared/llm/checkpoints/EleutherAI/pythia-160m \
        --out_dir out/lora/pythia-160m-finetuned
```

## Model Evaluation
Check the evaluation on the original model first:
```
salloc --gpus=1 llm-run.sh \
    python lit-gpt/eval/lm_eval_harness.py \
        --checkpoint_dir /home/shared/llm/checkpoints/EleutherAI/pythia-160m \
        --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" \
        --batch_size 4 \
        --save_filepath results-pythia-160m.json
```

Now for the fine-tuned model. Firest, we need to merge the weights when using lora for finetuning as [explained here](https://github.com/Lightning-AI/lit-gpt/blob/6178c7cc58ba82e5cce138e7a3159c384e2d3b0f/tutorials/finetune_lora.md). Then we can basically proceed in a similar manner as the original model, but be sure to check over the link and see if there is anything specific to your model that needs to be done. (For example, Llama 2 also needs a manual copy step for it's `tokenizer.model` file to complete the merge.)
```
mkdir -p out/lora_merged/pythia-160m-finetuned
salloc --gpus=1 llm-run.sh \
    python lit-gpt/scripts/merge_lora.py \
      --checkpoint_dir /home/shared/llm/checkpoints/EleutherAI/pythia-160m \
      --lora_path out/lora/pythia-160m-finetuned/lit_model_lora_finetuned.pth \
      --out_dir out/lora_merged/pythia-160m-finetuned
cp \
    /home/shared/llm/checkpoints/EleutherAI/pythia-160m/lit_config.json \
    /home/shared/llm/checkpoints/EleutherAI/pythia-160m/tokenizer_config.json \
    /home/shared/llm/checkpoints/EleutherAI/pythia-160m/tokenizer.json \
    out/lora_merged/pythia-160m-finetuned/.
```
and the final evaluation step as so:
```
salloc --gpus=1 llm-run.sh \
    python lit-gpt/eval/lm_eval_harness.py \
        --checkpoint_dir out/lora_merged/pythia-160m-finetuned \
        --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" \
        --batch_size 4 \
        --save_filepath results-pythia-160m-finetuned.json
```

## Multi-node GPU Training

Everything should run just fine for the `pythia-160m` model on a single 2080 Ti (12Gb) GPU, but if you increase the model size, batch size, or other settings then you may get a CUDA error saying you are out of GPU memory. This can generally happen when performing any of the main three stages of model use: inference, finetuning, or evaluation. Here I will assume inference and evaluation are fairly manageable for a larger model. For example, `pythia-410m` can be used in place of `pythia-160m` for all commands above, and the only step where you will run out of memory will be at the finetuning stage. That is, `pythia-410m` inference and evaluation will run fine on a single 2080 Ti, but it will fail to finetune on a single 2080 Ti. Let's solve this using the Lightning Fabric and distribute the model across two 2080 Ti GPUs instead.

If you open up the patched `lit-gpt/finetune/lora.py` script in your editor, you will find lines that use SLURM job environment variables to set `num_nodes` (using `SLURM_NNODES`) and `devices_per_node` (using `SLURM_GPUS_ON_NODE`). These could be hard-coded, changed to command-line arguments, or otherwise but this works well for us here since we are using SLURM.

So, to initiate multi-GPU finetuning then for this larger model, we can use:
```
sbatch --nodes=2 --gpus=2 llm-run.sh \
    python lit-gpt/finetune/lora.py \
        --data_dir /home/shared/llm/data/EleutherAI/pythia-410m/dolly \
        --checkpoint_dir /home/shared/llm/checkpoints/EleutherAI/pythia-410m \
        --out_dir out/lora/pythia-410m-finetuned
```

You will still need to merge the weights and copy other auxiliary files for this model, just as we did above for `pythia-160m` before evaluation. However, this completes the entire process for `pythia-410m` using two 2080 Ti GPs for distributed finetuning.

## Results for Pythia-160m finetuned on Dolly15k

Most of the results are statistically indistinguishable, but here they are:

<table border='1'>
  <tr>
    <th>Pythia-160m</th>
    <th>Pythia-160m (finetuned)</th>
  </tr>
  <tr>
    <td>
      <ul>
	<li>results
	  <ul>
	    <li>arithmetic_1dc
              <ul>
		<li>acc: 0.0</li>
		<li>acc_stderr: 0.0</li>
              </ul>
	    </li>
	    <li>openbookqa
              <ul>
		<li>acc: 0.162</li>
		<li>acc_stderr: 0.016494123566423515</li>
		<li>acc_norm: 0.27</li>
		<li>acc_norm_stderr: 0.019874354831287487</li>
              </ul>
	    </li>
	    <li>truthfulqa_mc
              <ul>
		<li>mc1: 0.24357405140758873</li>
		<li>mc1_stderr: 0.015026354824910782</li>
		<li>mc2: 0.445983277407095</li>
		<li>mc2_stderr: 0.014990213240210378</li>
              </ul>
	    </li>
	    <li>wikitext
              <ul>
		<li>word_perplexity: 33.44272972136349</li>
		<li>byte_perplexity: 1.9277559001908169</li>
		<li>bits_per_byte: 0.9469223835807006</li>
              </ul>
	    </li>
	  </ul>
	</li>
	<li>versions
	  <ul>
	    <li>arithmetic_1dc: 0</li>
	    <li>openbookqa: 0</li>
	    <li>truthfulqa_mc: 1</li>
	    <li>wikitext: 1</li>
	  </ul>
	</li>
	<li>config
	  <ul>
	    <li>model: pythia-160m</li>
	    <li>num_fewshot: 0</li>
	    <li>batch_size: 4</li>
	    <li>device: cuda:0</li>
	    <li>no_cache: True</li>
	    <li>limit: None</li>
	    <li>bootstrap_iters: 2</li>
	    <li>description_dict: None</li>
	  </ul>
	</li>
      </ul>
    </td>
    <td>
      <ul>
	<li>results
	  <ul>
	    <li>arithmetic_1dc
              <ul>
		<li>acc: 0.0025</li>
		<li>acc_stderr: 0.0011169148353275293</li>
              </ul>
	    </li>
	    <li>openbookqa
              <ul>
		<li>acc: 0.158</li>
		<li>acc_stderr: 0.016328049804579824</li>
		<li>acc_norm: 0.262</li>
		<li>acc_norm_stderr: 0.019684688820194716</li>
              </ul>
	    </li>
	    <li>truthfulqa_mc
              <ul>
		<li>mc1: 0.2460220318237454</li>
		<li>mc1_stderr: 0.015077219200662587</li>
		<li>mc2: 0.4328092115863148</li>
		<li>mc2_stderr: 0.01493920534206659</li>
              </ul>
	    </li>
	    <li>wikitext
              <ul>
		<li>word_perplexity: 33.43316373893594</li>
		<li>byte_perplexity: 1.9276527705228013</li>
		<li>bits_per_byte: 0.9468452012774183</li>
              </ul>
	    </li>
	  </ul>
	</li>
	<li>versions
	  <ul>
	    <li>arithmetic_1dc: 0</li>
	    <li>openbookqa: 0</li>
	    <li>truthfulqa_mc: 1</li>
	    <li>wikitext: 1</li>
	  </ul>
	</li>
	<li>config
	  <ul>
	    <li>model: pythia-160m</li>
	    <li>num_fewshot: 0</li>
	    <li>batch_size: 4</li>
	    <li>device: cuda:0</li>
	    <li>no_cache: True</li>
	    <li>limit: None</li>
	    <li>bootstrap_iters: 2</li>
	    <li>description_dict: None</li>
	  </ul>
	</li>
      </ul>
    </td>
  </tr>
</table>


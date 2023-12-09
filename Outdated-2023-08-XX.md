# llm-finetuning
A tutorial for fine-tuning LLMs on small cluster GPU hardware

# Overview

These are a set of instructions for downloading/training LLMs on our MTSU cluster resources. Most of the material is adapted from the [NeurIPS 2023 LLM Efficiency Challenge](https://github.com/ayulockin/neurips-llm-efficiency-challenge).

# Running on Babbage/Hamilton

You will need to have a working stack with PyTorch >= 2.1 since we will be utilizing [`lit-gpt`](https://github.com/Lightning-AI/lit-gpt) for our training framework which requires this minimal PyTorch version.

Custom stack is available on Babbage/Hamilton and compatible with [`apptainer`](https://apptainer.org/):
`/home/shared/sif/csci-2023-10-20.sif`

I recommend logging into our JupyterHub using [azuread](https://jupyterhub.cs.mtsu.edu/azuread/) then log into Babbage using a terminal and start a JupyterLab session using the stack above. I utilize `tmux` and the following command to start my job to get an A5000 on Babbage:
```
srun -G 1 -p research-gpu -t 2400 -x c[11-14] apptainer run --nv --env NB_UID=${UID} --env NB_USER=${USER} --env HF_HOME=${HOME}/hf_home --bind /home/shared/checkpoints:/home/checkpoints --writable-tmpfs -H jlab:${HOME} --env XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/pkgs/cuda-toolkit" --env NOTEBOOK_ARGS="--NotebookApp.base_url=/azuread/user/${USER}/proxy/absolute/9000/ --NotebookApp.custom_display_url=https://jupyterhub.cs.mtsu.edu" /home/shared/sif/csci-2023-10-20.sif
```
or if trying to get the A100 on Hamilton:
```

```

Once the job is in the queue and started up, I check which  node I am allocated using `squeue` and then log out and back in, adding the appropriate forwarding argument for that node (eg. `-L 9000:c17:8888` if I was allocated node `c17`). Note also that I have a subdirectory named `jlab` which I use for running the session: this just ensures I don't clobber any files in my home directory unintentionally.

# Installation of Packages

You will need to customize the environment a little:
```
pip install --user jsonargparse[signatures] sentencepiece bitsandbytes==0.41.0
MAX_JOBS=8 pip install --user flash-attn --no-build-isolation
```
Prep `lit-gpt` by pulling a version compatible with flash-attn 2. Note that the newest version dropped the flash-attn dependency, but then requires PyTorch 2.2. I will update this step when 2.2 rolls from nightly into stable.
```
git clone https://github.com/Lightning-AI/lit-gpt.git
cd lit-gpt
git checkout 6178c7cc58ba82e5cce138e7a3159c384e2d3b0f
cd ..
```

# Download/Prep Model

Note that I have downloaded some models already in `/home/shared/checkpoints`, so check there first before running the next command:
```
python lit-gpt/scripts/download.py --repo_id tiiuae/falcon-7b-instruct
```
You can use the pre-downloaded version instead here (or modify to point to your downloaded version):
```
python lit-gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct
```
Check that the model works:
```
python lit-gpt/generate/base.py --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct --prompt "Tell me an interesting fun fact about earth:"
```

# Prep Data
```
python lit-gpt/scripts/prepare_dolly.py --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct
```

# Fine-tune

```
python lit-gpt/finetune/lora.py --data_dir data/dolly --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct --precision bf16-true --out_dir out/lora/falcon-7b-instruct --quantize "bnb.nf4"
```
Double quantization - even less memory...
```
python lit-gpt/finetune/lora.py --data_dir data/dolly --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct --precision bf16-true --out_dir out/lora/falcon-7b-instruct --quantize "bnb.nf4-dq"
```

# Evaluation
Prep evaluation tools. Note that I have trouble running the fine-tuning process after the `lm-eval` package (below) is installed. You may need to temporarily uninstall that package to re-run the fine-tuning process. Then, you can reinstall `lm-eval` and perform evaluation.
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install --user --no-cache-dir -e .
cd ..
```
Check the evaluation on the original model:
```
python lit-gpt/eval/lm_eval_harness.py --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct --precision "bf16-true" --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" --batch_size 4 --save_filepath "results-falcon-7b-instruct.json"
```
Now for the fine-tuned model. There will be an issue at this stage where the `generate_prompt()` function cannot be imported due to version skew. You can open the `lit-gpt/eval/lm_eval_harness_lora.py` and comment out the offending line, then copy-paste the function from `lit-gpt/scripts/prepare_alpaca.py` directly below the commented line. (I'll work on performing these steps with the latest `lit-gpt` instead soon.)
```
python lit-gpt/eval/lm_eval_harness_lora.py --lora_path out/lora/falcon-7b-instruct/lit_model_lora_finetuned.pth --checkpoint_dir /home/checkpoints/tiiuae/falcon-7b-instruct --precision "bf16-true" --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" --batch_size 4 --save_filepath "results-falcon-7b-ft.json"
```

#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --time=0-10:00:00     # Requested time
#SBATCH --partition=research  # Partition/queue for running the job
#SBATCH --signal=SIGUSR1@90   # Enables auto-requeueing for SLURM/PyTorchLightning

# data locations
export LLM_DATA_DIR=/home/shared/llm
export LLM_CACHE_DIR=${HOME}/.cache/llm

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# Note that this is to match the arguments above here...
mkdir -p ${LLM_CACHE_DIR}
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

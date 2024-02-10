#!/usr/bin/env bash
#SBATCH --job-name=jupyter
#SBATCH --partition=slurm # You can pick from https://acg.maine.edu/hpc#h.b5slztm4yz12
#SBATCH --gres=gpu:1 # not clear if this is obeyed https://slurm.schedmd.com/gres.html
#SBATCH --time=02:00:00
#SBATCH --output=/home/anisha/Documents/PPGADev/jupyter.log

RUN_NAME="enjoy_ppga_script"
eval "$(conda shell.bash hook)"
conda activate ppga
echo $RUN_NAME
srun echo -ne "anishapv\n" | python -m algorithm.enjoy_ppga_script

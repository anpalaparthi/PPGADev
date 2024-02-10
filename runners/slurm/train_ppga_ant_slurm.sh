#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=/home/anisha/Documents/PPGADev/anisha_out.txt
#SBATCH --error=/home/anisha/Documents/PPGADev/anisha_error.txt
ENV_NAME="ant"
GRID_SIZE=10  # number of cells per archive dimension
SEED=1111


RUN_NAME="paper_ppga_"$ENV_NAME"_seed_"$SEED
eval "$(conda shell.bash hook)"
conda activate ppga
echo $RUN_NAME
export WANDB_API_KEY=891d210ba7d22f573c05eab8fee22c8ff1e662bc
srun echo -ne "anishapv\n" | python -m algorithm.train_ppga --env_name=$ENV_NAME \
                                     --rollout_length=128 \
                                     --use_wandb=True \
                                     --seed=$SEED \
                                     --wandb_group=paper_v2 \
                                     --num_dims=4 \
                                     --num_minibatches=8 \
                                     --update_epochs=4 \
                                     --normalize_obs=True \
                                     --normalize_returns=True \
                                     --wandb_run_name=$RUN_NAME \
                                     --popsize=300 \
                                     --env_batch_size=3000 \
                                     --learning_rate=0.001 \
                                     --vf_coef=2 \
                                     --max_grad_norm=1 \
                                     --torch_deterministic=False \
                                     --total_iterations=2000 \
                                     --dqd_algorithm=cma_maega \
                                     --calc_gradient_iters=10 \
                                     --move_mean_iters=10 \
                                     --archive_lr=0.1 \
                                     --restart_rule=no_improvement \
                                     --sigma0=3.0 \
                                     --threshold_min=-500 \
                                     --grid_size=$GRID_SIZE \
                                     --expdir=./experiments/paper_ppga_"$ENV_NAME"

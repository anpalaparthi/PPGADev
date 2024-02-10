#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=/home/anisha/Documents/PPGADev/anisha_ppo_ant_energy.txt
#SBATCH --error=/home/anisha/Documents/PPGADev/anisha_ppo_ant_energy_.txt
ENV_NAME="ant"
SEED=1111


RUN_NAME="ppo_energy_"$ENV_NAME"_seed_"$SEED
eval "$(conda shell.bash hook)"
conda activate ppga
echo $RUN_NAME
export WANDB_API_KEY=891d210ba7d22f573c05eab8fee22c8ff1e662bc
srun echo -ne "anishapv\n" | python -u -m RL.train_ppo --env_name=$ENV_NAME \
                                     --rollout_length=128 \
                                     --use_wandb=True \
                                     --seed=$SEED \
                                     --wandb_group=qdrl \
                                     --num_minibatches=8 \
                                     --update_epochs=4 \
                                     --total_timesteps=1_000_000_000 \
				     --env_type=brax \
				     --num_dims=5 \
                                     --normalize_obs=True \
                                     --normalize_returns=True \
                                     --wandb_run_name=$RUN_NAME \
                                     --env_batch_size=3000 \
                                     --learning_rate=0.001 \
                                     --vf_coef=2 \
                                     --max_grad_norm=1 \
                                     --torch_deterministic=False \
                                     --is_energy_measures=True

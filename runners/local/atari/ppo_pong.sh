#!/usr/bin/env bash

SEED=1111
RUN_NAME="ppo_pong_seed_"$SEED
echo $RUN_NAME

python -u -m RL.train_ppo --env_name=pong \
                          --env_type=gym \
                          --use_wandb=True\
                          --wandb_group=qdrl \
                          --normalize_obs=False \
                          --normalize_returns=True \
                          --num_dims=0 \
                          --wandb_run_name=$RUN_NAME \
                          --is_energy_measures=False \
                          --rollout_length=128 \
                          --clip_coef=0.1  \
                          --clip_vloss=True \
                          --env_batch_size=16 \
                          --num_minibatches=4 \
                          --update_epochs=4 \
                          --learning_rate=0.00025 \
                          --norm_adv=True \
                          --seed=1111 \
                          --total_timesteps=10_000_000 \
                          --torch_deterministic=True \
                          --entropy_coef=0.01 \
                          --vf_coef=0.5 \
                          --max_grad_norm=0.5
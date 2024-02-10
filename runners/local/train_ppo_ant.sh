#!/usr/bin/env bash

ENV_NAME="ant"
SEED=1111


RUN_NAME="ppo_energy_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -u -m RL.train_ppo --env_name=$ENV_NAME \
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

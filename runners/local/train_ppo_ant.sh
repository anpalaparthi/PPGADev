#!/usr/bin/env bash

ENV_NAME="ant"
SEED=1111


RUN_NAME="ppo_energy_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -u -m algorithm.train_ppga --env_name=$ENV_NAME \
                                     --rollout_length=128 \
                                     --use_wandb=True \
                                     --seed=$SEED \
                                     --wandb_group=qdrl \
                                     --num_minibatches=8 \
                                     --update_epochs=4 \
                                     --normalize_obs=True \
                                     --normalize_returns=True \
                                     --wandb_run_name=$RUN_NAME \
                                     --env_batch_size=3000 \
                                     --learning_rate=0.001 \
                                     --vf_coef=2 \
                                     --max_grad_norm=1 \
                                     --torch_deterministic=False \
                                     --is_energy_measures=True \
                                     --expdir=./experiments/paper_ppga_"$ENV_NAME"

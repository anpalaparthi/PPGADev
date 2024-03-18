#!/usr/bin/env bash

SEED=0
RUN_NAME="ppga_space_invaders_seed"$SEED
echo $RUN_NAME

python -u -m algorithms.train_ppga --env_name=ALE/SpaceInvaders-v5 \
                                    --env_type=gym \
                                    --use_wandb=False \
                                    --wandb_group=atari \
                                    --normalize_obs=False \
                                    --normalize_returns=True \
                                    --rollout_length=128 \
                                    --num_dims=1 \
                                    --grid_size=100 \
                                    --popsize=16 \
                                    --restart_rule=no_improvement \
                                    --sigma0=0.5 \
                                    --calc_gradient_iters=10 \
                                    --move_mean_iters=10 \
                                    --archive_lr=0.1 \
                                    --threshold_min=0 \
                                    --save_scheduler=False \
                                    --seed=0 \
                                    --rollout_length=128 \
                                    --clip_coef=0.1 \
                                    --clip_vloss=True \
                                    --env_batch_size=16 \
                                    --num_minibatches=4 \
                                    --update_epochs=4 \
                                    --learning_rate=0.00025 \
                                    --norm_adv=True \
                                    --torch_deterministic=True \
                                    --entropy_coef=0.01 \
                                    --vf_coef=0.5 \
                                    --max_grad_norm=0.5 \
                                    --expdir=experiments/ppga_space_invaders \
                                    --log_arch_freq=1
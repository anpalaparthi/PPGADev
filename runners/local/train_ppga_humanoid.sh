#!/usr/bin/env bash

ENV_NAME="humanoid"
GRID_SIZE=20  # number of cells per archive dimension
SEED=1111


RUN_NAME="paper_ppga_energy_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -u -m algorithm.train_ppga --env_name=$ENV_NAME \
                                --rollout_length=128 \
                                --use_wandb=True \
                                --wandb_group=paper \
                                --num_dims=3 \
                                --seed=$SEED \
                                --anneal_lr=False \
                                --num_minibatches=8 \
                                --update_epochs=4 \
                                --normalize_obs=True \
                                --normalize_returns=True \
                                --clip_obs_rew=True \
                                --adaptive_stddev=False \
                                --clip_obs_rew=True \
                                --wandb_run_name=$RUN_NAME\
                                --popsize=300 \
                                --env_batch_size=3000 \
                                --learning_rate=0.0003 \
                                --vf_coef=2 \
                                --entropy_coef=0.0 \
                                --target_kl=0.008 \
                                --max_grad_norm=1 \
                                --total_iterations=2000 \
                                --dqd_algorithm=cma_maega \
                                --sigma0=0.5 \
                                --restart_rule=no_improvement \
                                --calc_gradient_iters=10 \
                                --move_mean_iters=10 \
                                --archive_lr=0.1 \
                                --threshold_min=-180 \
                                --grid_size=$GRID_SIZE \
                                --take_archive_snapshots=True \
                                --use_cvt_archive=True \
                                --cvt_cells=1000 \
                                --cvt_samples=100000 \
                                --cvt_use_kd_tree=True \
                                --is_energy_measures=True \
                                --expdir=./experiments/paper_ppga_"$ENV_NAME"

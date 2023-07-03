#!/usr/bin/env bash

ENV_NAME="ant"
GRID_SIZE=7  # number of cells per archive dimension
SEED=1111


RUN_NAME="energy_1000_paper_ppga_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -m algorithm.train_ppga --env_name=$ENV_NAME \
                                     --rollout_length=128 \
                                     --use_wandb=True \
                                     --seed=$SEED \
                                     --wandb_group=paper \
                                     --num_dims=5 \
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
                                     --take_archive_snapshots=True \
                                     --use_cvt_archive=True \
                                     --cvt_cells=1000 \
                                     --cvt_samples=100000 \
                                     --cvt_use_kd_tree=True \
                                     --is_energy_measures=True \
                                     --expdir=./experiments/paper_ppga_"$ENV_NAME"

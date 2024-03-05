import argparse
import os
import sys
import time
from distutils.util import strtobool
from attrdict import AttrDict
from utils.utilities import config_wandb, log
from RL.ppo import PPO

from models.actor_critic import Actor
from ribs.archives import CVTArchive, GridArchive
from envs.gym_atari.atari_env import make_vec_env as make_vec_env_gym
from envs.envpool.envpool_env import make_env as make_envpool_env

# envpool imports
import gymnasium as gym
import envpool
import numpy as np
from packaging import version

import torch
import cv2
from collections import deque
from collections import defaultdict

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
print("os env = ", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


class RecordEpisodeStatistics(gym.Wrapper):

  def __init__(self, env, deque_size=100):
    super(RecordEpisodeStatistics, self).__init__(env)
    self.num_envs = getattr(env, "num_envs", 1)
    self.episode_returns = None
    self.episode_lengths = None
    # get if the env has lives
    self.has_lives = False
    env.reset()
    info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
    if info["lives"].sum() > 0:
      self.has_lives = True
      print("env has lives")

  def reset(self, **kwargs):
    if is_legacy_gym:
      observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
    else:
      observations, _ = super(RecordEpisodeStatistics, self).reset(**kwargs)
    self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    self.lives = np.zeros(self.num_envs, dtype=np.int32)
    self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    return observations

  def step(self, action):
    if is_legacy_gym:
      observations, rewards, dones, infos = super(
        RecordEpisodeStatistics, self
      ).step(action)
    else:
      observations, rewards, term, trunc, infos = super(
        RecordEpisodeStatistics, self
      ).step(action)
      dones = term + trunc
    self.episode_returns += infos["reward"]
    self.episode_lengths += 1
    self.returned_episode_returns[:] = self.episode_returns
    self.returned_episode_lengths[:] = self.episode_lengths
    all_lives_exhausted = infos["lives"] == 0
    if self.has_lives:
      self.episode_returns *= 1 - all_lives_exhausted
      self.episode_lengths *= 1 - all_lives_exhausted
    else:
      self.episode_returns *= 1 - dones
      self.episode_lengths *= 1 - dones
    infos["r"] = self.returned_episode_returns
    infos["l"] = self.returned_episode_lengths
    return (
      observations,
      rewards,
      dones,
      infos,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)
    parser.add_argument('--wandb_project', type=str, default='PPGA')
    parser.add_argument('--report_interval', type=int, default=5, help='Log objective results every N updates')

    # algorithm args
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--env_type', type=str, choices=['brax', 'isaac', 'envpool', 'gym'], help='Whether to use cpu-envs or gpu-envs for rollouts')
    # args for brax
    parser.add_argument('--env_batch_size', default=1, type=int, help='Number of parallel environments to run')

    # args for cpu-envs
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes to spawn. '
                             'Should always be <= number of logical cores on your machine')
    parser.add_argument('--envs_per_worker', type=int, default=1,
                        help='Num envs each worker process will step through sequentially')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
    # ppo hyperparams
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda discount used for general advantage est')
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--update_epochs', type=int, default=10, help='The K epochs to update the policy')
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_value_coef", type=float, default=0.2,
                        help="value clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument('--normalize_obs', type=lambda x: bool(strtobool(x)), default=False, help='Normalize observations across a batch using running mean and stddev')
    parser.add_argument('--normalize_returns', type=lambda x: bool(strtobool(x)), default=False, help='Normalize rewards across a batch using running mean and stddev')
    parser.add_argument('--value_bootstrap', type=lambda x: bool(strtobool(x)), default=False, help='Use bootstrap value estimates')

    parser.add_argument('--weight_decay', type=float, default=None, help='Apply L2 weight regularization to the NNs')
    # vestigial QD params
    parser.add_argument('--num_dims', type=int)
    parser.add_argument('--is_energy_measures', type=lambda x: bool(strtobool(x)), default=False,
                        help="use the wrapper with only forward reward and control cost part of measures")
    parser.add_argument('--clip_obs_rew', type=lambda x: bool(strtobool(x)), default=False, help='Clip obs and rewards b/w -10 and 10')

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


if __name__ == '__main__':
    force_cudnn_initialization()
    cfg = parse_args()

    if cfg.seed is None:
        cfg.seed = int(time.time()) + int(os.getpid())

    if cfg.env_type == 'brax':
        from envs.brax_custom.brax_env import make_vec_env_brax
        vec_env = make_vec_env_brax(cfg)
    elif cfg.env_type == 'envpool':  # add support for envpool environments
        print("ENV ENV POOL POOL")
        # num_envs_envpool = 8
        vec_env = make_envpool_env(cfg)
        vec_env.num_envs = int(cfg.env_batch_size)
        vec_env.single_action_space = vec_env.action_space
        print(f"action space orig: {vec_env.action_space}, {vec_env.action_space.shape}")
        vec_env.single_observation_space = vec_env.observation_space
        # vec_env = RecordEpisodeStatistics(vec_env)
        # TODO: maybe add record episode statistics
    elif cfg.env_type == 'gym':
        vec_env = make_vec_env_gym(cfg)
    else:
        print("NOT ANYTHING")
        raise NotImplementedError(f'{cfg.env_type} is undefined for "env_type"')

    print("is energy measures = ", cfg.is_energy_measures)
    if cfg.is_energy_measures:
        print("vec env: ")
        print(vec_env)
    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)
    # if cfg.env_type == 'envpool':
    #     cfg.num_envs = num_envs_envpool
    cfg.num_emitters = 1
    cfg.envs_per_model = cfg.num_envs // cfg.num_emitters
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    print(f"obs shape: {cfg.obs_shape}")
    if cfg.env_type in ['envpool', 'gym']:
        cfg.single_action_space = vec_env.single_action_space
    # else:
    cfg.action_shape = vec_env.single_action_space.shape
    print(f"cfg action shape: {cfg.action_shape}")

    log.debug(f'Environment: {cfg.env_name}, obs_shape: {cfg.obs_shape}, action_shape: {cfg.action_shape}')

    if cfg.use_wandb:
        config_wandb(cfg=cfg, batch_size=cfg.batch_size, total_steps=cfg.total_timesteps, run_name=cfg.wandb_run_name, wandb_group=cfg.wandb_group, wandb_project=cfg.wandb_project)

    alg = PPO(cfg)
    num_updates = cfg.total_timesteps // cfg.batch_size
    alg.train(vec_env, num_updates, rollout_length=cfg.rollout_length)

    sys.exit(0)
	# TODO: add envpool environments

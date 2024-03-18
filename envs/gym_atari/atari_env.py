import gymnasium as gym
import ale_py
import torch

from envs.gym_atari.wrappers.measures import *
from envs.wrappers.common_wrappers import TorchWrapper, VecResizeObservation
from envs.gym_atari.wrappers.image import PixelNormalization
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

_atari_info_wrappers = {
    'ALE/Asteroids-v5': AsteroidsInfoWrapper,
    'ALE/Tennis-v5': TennisInfoWrapper,
    'ALE/SpaceInvaders-v5': SpaceInvadersInfoWrapper
}


def make_env(env_name):
    def thunk():
        env = gym.make(env_name,
                       render_mode='rgb_array',
                       frameskip=1,
                       repeat_action_probability=0.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = _atari_info_wrappers[env_name](env)
        return env

    return thunk


def make_vec_env(cfg):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg.env_name) for _ in range(int(cfg.env_batch_size))]
    )
    envs = TorchWrapper(envs, device='cuda' if torch.cuda.is_available() else 'cpu')
    envs = VecResizeObservation(envs, (84, 84))
    envs = PixelNormalization(envs)
    return envs
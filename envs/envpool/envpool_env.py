import envpool
import torch

from attrdict import AttrDict
from envs.wrappers.common_wrappers import EnvPoolTorchWrapper
from envs.gym_atari.wrappers.image import PixelNormalization


def make_env(cfg: AttrDict):
    env = envpool.make(task_id=cfg.env_name,
                       env_type='gym',
                       num_envs=int(cfg.env_batch_size),
                       episodic_life=True,
                       reward_clip=True)
    env = EnvPoolTorchWrapper(env, device='cpu')
    env = PixelNormalization(env)
    return env

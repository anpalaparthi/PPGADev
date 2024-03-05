import envpool
import torch

from attrdict import AttrDict
from envs.wrappers.common_wrappers import EnvPoolTorchWrapper


def make_env(cfg: AttrDict):
    env = envpool.make(task_id=cfg.env_name,
                       env_type='gym',
                       num_envs=int(cfg.env_batch_size),
                       episodic_life=True,
                       reward_clip=False)
    env = EnvPoolTorchWrapper(env, device='cpu')
    return env

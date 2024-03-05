import gymnasium as gym
import ale_py
import numpy as np

from time import time
from envs.gym_atari.atari_env import make_vec_env
from attrdict import AttrDict


def main():
    cfg = AttrDict({'env_name': 'ALE/Tennis-v5', 'env_batch_size': 16})
    env = make_vec_env(cfg)
    # env = gym.vector.make('ALE/Tennis-v5', num_envs=16)
    # test fps
    steps = 1000
    env.reset()
    start = time()
    for _ in range(steps):
        act = np.random.randint(0, 3, size=env.action_space.shape)
        obs, reward, done, trunc, info = env.step(act)

    elapsed = time() - start
    fps = steps / elapsed
    print(f'FPS: {fps:.2f}')


if __name__ == "__main__":
    main()

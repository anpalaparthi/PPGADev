import gymnasium as gym
import gym as g
import torch
import numpy as np

from gymnasium.error import DependencyNotInstalled
from torchvision.transforms import Resize


class TorchWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env, device: str):
        super(TorchWrapper, self).__init__(env)
        self.device = device

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = torch.from_numpy(obs).to(self.device).to(torch.float32)
        reward = torch.from_numpy(reward).to(self.device)
        trunc = torch.from_numpy(trunc).to(self.device)
        return obs, reward, done, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = torch.from_numpy(np.array(obs)).to(self.device).to(torch.float32)
        return obs, info


class EnvPoolTorchWrapper(g.Wrapper):
    def __init__(self, env, device: str):
        super(EnvPoolTorchWrapper, self).__init__(env)
        self.device = device

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = torch.from_numpy(obs).to(self.device).to(torch.float32)
        reward = torch.from_numpy(reward).to(self.device)
        trunc = torch.from_numpy(trunc).to(self.device)
        return obs, reward, done, trunc, info

    def reset(self):
        obs, info = self.env.reset()
        obs = torch.from_numpy(np.array(obs)).to(self.device).to(torch.float32)
        return obs, info


class VecResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape) -> None:
        '''
        Vectorized resize atari image observations to be of shape "shape"
        '''
        gym.ObservationWrapper.__init__(self, env)

        self.resize = Resize(shape)
        obs_shape = (env.single_observation_space.shape[0],) + shape
        self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        vec_obs_shape = (env.observation_space.shape[0],) + obs_shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=vec_obs_shape)

    def observation(self, observation: torch.Tensor):
        '''
        Resize a batch of grayscale images
        Args:
            observation: batch of images with dims (BATCH_SIZE, FRAMES, WIDTH, HEIGHT)
        '''

        observation = self.resize(observation)
        return observation

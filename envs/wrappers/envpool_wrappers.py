import gym
import torch


class EnvPoolTorchWrapper(gym.Wrapper):
    def __init__(self, env, device: str):
        super(EnvPoolTorchWrapper, self).__init__(env)
        self.device = device

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(obs).to(self.device)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).to(self.device)
        return obs
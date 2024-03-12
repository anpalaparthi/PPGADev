import gym
import torch


class PixelNormalization(gym.ObservationWrapper):
    def observation(self, obs: torch.Tensor):
        return (obs / 255.0).mul(2.0).sub(1.0).clamp(-1., 1.)


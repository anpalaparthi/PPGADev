import gymnasium as gym
import torch
import numpy as np

# https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
# see this link for the mapping from ram to game extra stats and info


class SpaceInvadersInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SpaceInvadersInfoWrapper, self).__init__(env)
        # player absolute velocity as measure
        self.player_x_prev = None

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)

        ram = self.ale.getRAM().astype(np.float32)

        if self.player_x_prev is None:
            self.player_x_prev = ram[28]

        x_vel = ram[28] - self.player_x_prev
        info['measures'] = abs(x_vel)
        self.player_x_prev = ram[28]

        return obs, reward, done, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.player_x_prev = self.ale.getRAM().astype(np.float32)[28]
        info['measures'] = 0.0
        return obs, info


class TennisInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TennisInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        ram = self.ale.getRAM()
        info['player_x'] = ram[26]
        info['player_y'] = ram[24]


        return obs, reward, done, trunc, info


class AsteroidsInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AsteroidsInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        ram = self.ale.getRAM()
        info['player_x'] = ram[73]
        info['player_y'] = ram[74]

        return obs, reward, done, trunc, info


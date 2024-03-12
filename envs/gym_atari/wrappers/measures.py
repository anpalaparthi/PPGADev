import gymnasium as gym

# https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
# see this link for the mapping from ram to game extra stats and info


class SpaceInvadersInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SpaceInvadersInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        ram = self.ale.getRAM()
        info['player_x'] = ram[28]

        return obs, reward, done, trunc, info


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


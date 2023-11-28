import torch
import torch.nn as nn
import numpy as np

from models.policy import StochasticPolicy
from abc import ABC, abstractmethod
from typing import List
from utils.normalize import ReturnNormalizer, ObsNormalizer
from torch.amp import autocast

from torch.distributions.categorical import Categorical


class VectorizedLinearBlock(nn.Module):
    def __init__(self, weights: torch.Tensor, biases=None, device=None, dtype=None, env_type="brax") -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(weights).to(self.device)  # one slice of all the mlps we want to process as a batch
        self.bias = nn.Parameter(biases).to(self.device) if biases is not None else None
        self.env_type = env_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_per_weight = x.shape[0] // self.weight.shape[0]
        #print("x.shape")
        #print(x.shape)
        #print("self.weight")
        #print(self.weight.shape)
        #print("obs per weight")
        #print(obs_per_weight)
        # print("x.shape = ", x.shape)
        # print("len(x.shape) = ", len(x.shape))
        # print("x.shape[0] = ", x.shape[0])
        # print("x.shape[1] = ", x.shape[1])
        # print("weight.shape = ", self.weight.shape)
        if ((self.env_type == "envpool") and (len(x.shape) == 4)):
            x = torch.reshape(x, (1, -1, x.shape[1] * x.shape[2] * x.shape[3]))
        elif ((self.env_type == "envpool") and (len(x.shape) == 5)):
            x = torch.reshape(x, (1, -1, x.shape[2] * x.shape[3] * x.shape[4]))
        else:
            x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weight, 1, 2).to(self.device)
        # print("after reshape x.shape = ", x.shape)
        # print("after reshape wt.shape = ", w_t.shape)
        with autocast(device_type=self.device.type):
            y = torch.bmm(x, w_t)
        if self.bias is not None:
            y = torch.transpose(y, 0, 1)
            y += self.bias

        out_features = self.weight.shape[1]
        y = torch.transpose(y, 0, 1)
        y = torch.reshape(y, shape=(-1, out_features))
        return y


class VectorizedPolicy(StochasticPolicy, ABC):
    def __init__(self, models, model_fn, obs_shape, action_shape, normalize_obs=False, normalize_returns=False):
        StochasticPolicy.__init__(self, normalize_obs=normalize_obs, obs_shape=obs_shape, normalize_returns=normalize_returns)
        if not isinstance(models, np.ndarray):
            models = np.array(models)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_models = len(models)
        self.model_fn = model_fn
        self.blocks: List[VectorizedLinearBlock]
        self.actor_mean: nn.Sequential
        self.actor_logstd: nn.Parameter
        self.normalize_obs = normalize_obs
        self.normalize_returns = normalize_returns
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        if normalize_obs:
            self.obs_normalizers = [model.obs_normalizer for model in models]
        if normalize_returns:
            self.rew_normalizers = [model.return_normalizer for model in models]

    def _vectorize_layers(self, layer_name, models, env_type="brax"):
        '''
        Vectorize a specific nn.Sequential list of layers across all models of homogenous architecture
        :param layer_name: name of a nn.Sequential block
        :return: vectorized nn.Sequential block
        '''
        assert hasattr(models[0], layer_name), f'{layer_name=} not in the model'
        all_models_layers = [getattr(models[i], layer_name) for i in range(self.num_models)]
        num_layers = len(getattr(models[0], layer_name))
        # print("num layers")
        # print(num_layers)
        # print("num models")
        # print(self.num_models)
        # print("all_models_layers")
        # print(all_models_layers)
        blocks = []
        for i in range(0, num_layers):
            if not isinstance(all_models_layers[0][i], nn.Linear):
                continue
            weights_slice = [all_models_layers[j][i].weight.to(self.device) for j in range(self.num_models)]
            bias_slice = [all_models_layers[j][i].bias.to(self.device) for j in range(self.num_models)]

            # print("weights slice shape before")
            # print(len(weights_slice))
            # print(weights_slice[0].shape)
            weights_slice = torch.stack(weights_slice)
            # print("weights slice shape after")
            # print(weights_slice.shape)
            bias_slice = torch.stack(bias_slice)
            nonlinear = all_models_layers[0][i + 1] if i + 1 < num_layers else None
            block = VectorizedLinearBlock(weights_slice, bias_slice, env_type=env_type)
            blocks.append(block)
            if nonlinear is not None:
                blocks.append(nonlinear)
        return blocks

    def vec_to_models(self):
        '''
        Returns a list of models view of the object
        '''
        models = [self.model_fn(self.obs_shape, self.action_shape, self.normalize_obs, self.normalize_returns)
                  for _ in range(self.num_models)]
        for i, model in enumerate(models):
            for l, layer in enumerate(self.actor_mean):
                # layer could be a nonlinearity
                if not isinstance(layer, VectorizedLinearBlock):
                    continue
                model.actor_mean[l].weight.data = layer.weight.data[i]
                model.actor_mean[l].bias.data = layer.bias.data[i]

            # update obs/rew normalizers
            if self.normalize_obs:
                model.obs_normalizer = self.obs_normalizers[i]
            if self.normalize_returns:
                model.return_normalizer = self.rew_normalizers[i]

            # update action logprobs
            model.actor_logstd.data = self.actor_logstd[i]
        return models

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_action(self, obs, action=None):
        pass

    # default env_type = not 'envpool'
    def vec_normalize_obs(self, obs, env_type="brax"):
        # TODO: make this properly vectorized
        # print("vec normalize obs.shape = ", obs.shape)
        if (env_type != 'envpool'):
            obs = obs.reshape(self.num_models, obs.shape[0] // self.num_models, -1)
        # print("vec normalize obs.reshape = ", obs.shape)
        for i, (model_obs, normalizer) in enumerate(zip(obs, self.obs_normalizers)):
            obs[i] = normalizer(model_obs)
        
        if (env_type != 'envpool'):
            return obs.reshape(-1, obs.shape[-1])
        else:
            return obs

    def vec_normalize_returns(self, rewards):
        # TODO: make this properly vectorized
        num_envs = rewards.shape[0]
        envs_per_model = num_envs // self.num_models
        rewards = rewards.reshape(self.num_models, envs_per_model)
        for i, (model_rews, normalizer) in enumerate(zip(rewards, self.rew_normalizers)):
            rewards[i] = normalizer(model_rews)
        return rewards.reshape(-1)


class VectorizedActor(VectorizedPolicy):
    def __init__(self, models, model_fn, obs_shape, action_shape, normalize_obs=False, normalize_returns=False, env_type="brax"):
        VectorizedPolicy.__init__(self, models, model_fn, obs_shape, action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_returns)
        self.blocks = self._vectorize_layers('actor_mean', models, env_type=env_type)
        self.actor_mean = nn.Sequential(*self.blocks)
        if (env_type != 'envpool'):
            action_logprobs = [model.actor_logstd for model in models]
            action_logprobs = torch.cat(action_logprobs).to(self.device)
            self.actor_logstd = nn.Parameter(action_logprobs)
        self.env_type = env_type

    @autocast(device_type='cuda')
    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None):
        # print("self.actor_mean.type = ", type(self.actor_mean))
        # print("get action obs.shape = ", obs.shape)
        if (self.env_type == 'envpool'):
            logits = self.actor_mean(obs)
            probs = Categorical(logits=logits)
        else:      
            with autocast(device_type=self.device.type):
                action_mean = self.actor_mean(obs)
            repeats = obs.shape[0] // self.num_models
            action_logstd = torch.repeat_interleave(self.actor_logstd, repeats, dim=0)
            action_logstd = action_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            #print("action mean: ")
            #print(action_mean)
            probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()

        if (self.env_type == 'envpool'):
            return action, probs.log_prob(action), probs.entropy() 
        else:
            return action, probs.log_prob(action).sum(1), probs.entropy()

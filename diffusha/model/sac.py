import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self,obs_space, act_space,goal_dim = None):
        super().__init__()
        self.goal_dim = goal_dim
        if goal_dim > 0:
            self.fc1 = nn.Linear(np.array(obs_space.shape).prod() + np.prod(act_space.shape) + goal_dim, 256)
        else:
            self.fc1 = nn.Linear(np.array(obs_space.shape).prod() + np.prod(act_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a, g=None):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, a, g], 1)
        else:
            x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_space,act_space,goal_dim=None):
        super().__init__()
        self.goal_dim = goal_dim
        if goal_dim > 0:
            self.fc1 = nn.Linear(np.array(obs_space.shape).prod() + goal_dim, 256)
        else:
            self.fc1 = nn.Linear(np.array(obs_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(act_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(act_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((act_space.high -act_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((act_space.high + act_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x,g=None):
        if self.goal_dim >0 and g is not None:
            x = torch.cat([x, g], 1)
        else:
            x = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x,g,deterministic=False):
        mean, log_std = self(x,g)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if deterministic:
            return mean
        return action, log_prob, mean
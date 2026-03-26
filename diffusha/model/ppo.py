import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


        



        
class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space,goal_dim = None,):
        super().__init__()
        self.goal_dim = goal_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod() + goal_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod() + goal_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(act_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(act_space.shape)) * -0.5)

    def get_value(self, x,g = None):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, g],-1)
        return self.critic(x).squeeze(-1)
    
    def get_action(self, x,g = None, deterministic=False):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, g],-1)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, g = None, action=None):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, g],-1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_space, act_space,goal_dim = None,):
        super().__init__()
        self.goal_dim = goal_dim
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod() + goal_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(act_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(act_space.shape)) * -0.5)

    def get_action(self, x,g = None, deterministic=False):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, g],-1)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    
    def get_action_dist(self, x, g = None, action=None):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, g],-1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

class Critic(nn.Module):
    def __init__(self, obs_space, act_space,goal_dim = None,):
        super().__init__()
        self.goal_dim = goal_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod() + goal_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
    def get_value(self, x,g = None):
        if self.goal_dim > 0 and g is not None:
            x = torch.cat([x, g],-1)
        return self.critic(x).squeeze(-1)

class ActorCritic_v2(nn.Module):
    def __init__(self, obs_space, act_space,goal_dim = None,):
        super().__init__()
        self.goal_dim = goal_dim
        self.actor = Actor(obs_space,act_space,goal_dim)
        self.critic = Critic(obs_space,act_space,goal_dim)

    def get_value(self, x,g = None):
        return self.critic.get_value(x,g)
    
    def get_action(self, x,g = None, deterministic=False):
        return self.actor.get_action(x,g,deterministic)
    
    def get_action_and_value(self, x, g = None, action=None):
        action, probs, ent = self.actor.get_action_dist(x,g,action)
        value = self.critic.get_value(x,g)
        return action,probs,ent,value


class Actor_v2(nn.Module):
    def __init__(self, obs_space, act_space, goal_dim=None,action_rescale = 0.3):
        super().__init__()
        self.goal_dim = goal_dim or 0
        self.action_rescale = action_rescale 
        
        obs_dim = np.prod(obs_space.shape)
        act_dim = np.prod(act_space.shape)

        # Store action bounds as buffers so they're on the same device:
        action_low = act_space.low
        action_high = act_space.high
        action_low[0:-1] *= self.action_rescale
        action_high[0:-1] *= self.action_rescale
        self.register_buffer("action_low",  torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        # print("Debug: ", self.action_low, self.action_high)
        # Define your mean network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim + self.goal_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01*np.sqrt(2)),
        )
        # Define log-std as a parameter
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs, g=None, deterministic=False):
        """ 
        Return an *already-squashed-and-bounded* action in [action_low, action_high].
        """
        device = next(self.actor_mean.parameters()).device
        obs = obs.to(device)
        g = g.to(device) if g is not None else None
        if self.goal_dim > 0 and g is not None:
            obs = torch.cat([obs, g], dim=-1)

        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)  # shape: [batch, act_dim]

        # Construct the distribution with the Tanh + Affine transforms
        loc = 0.5 * (self.action_high + self.action_low)
        scale = 0.5 * (self.action_high - self.action_low)
        
        base_dist = Normal(mean, std)
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=loc, scale=scale)]
        dist = TransformedDistribution(base_dist, transforms)

        if deterministic:
            # In practice, you might just take 'mean' -> Tanh -> scale if you want a deterministic action.
            # or if you want the "mode" under the Tanh transform, you'd do:
            # z_mode = mean  (since normal's mode = mean)
            # a_mode = transforms(...).
            z_mode = mean
            # We apply .tanh -> .affine manually for the mode:
            squashed = torch.tanh(z_mode)
            action = loc + scale * squashed
        else:
            action = dist.rsample()  # reparameterized sample

        return action

    def get_action_dist(self, obs, g=None, action=None):
        """
        Return the distribution, so we can get log_prob, entropy, etc.
        If `action` is provided, we compute the log_prob of that action under the distribution.
        """
        device = next(self.actor_mean.parameters()).device
        obs = obs.to(device)
        g = g.to(device) if g is not None else None

        if self.goal_dim > 0 and g is not None:
            obs = torch.cat([obs, g], dim=-1)

        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)

        loc = 0.5 * (self.action_high + self.action_low)
        scale = 0.5 * (self.action_high - self.action_low)
        
        base_dist = Normal(mean, std)
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=loc, scale=scale)]
        dist = TransformedDistribution(base_dist, transforms)

        if action is None:
            # sample from the distribution
            action = dist.rsample()

        # log_prob and entropy 
        log_prob = dist.log_prob(action).sum(-1)  # sum across action_dim
        # entropy = dist.entropy().sum(-1)          # also sum across action_dim
        entropy = -log_prob.sum(-1)
        entropy = entropy.mean(0)

        return action, log_prob, entropy
    
    def get_action(self,obs, g = None, deterministic = True):
        return self.forward(obs,g,deterministic)



class ActorCritic_v3(nn.Module):
    def __init__(self, obs_space, act_space,goal_dim = None,action_rescale = 0.3):
        super().__init__()
        self.goal_dim = goal_dim
        self.actor = Actor_v2(obs_space,act_space,goal_dim,action_rescale)
        self.critic = Critic(obs_space,act_space,goal_dim)

    def get_value(self, x,g = None):
        return self.critic.get_value(x,g)
    
    def get_action(self, x,g = None, deterministic=False):
        return self.actor.get_action(x,g,deterministic)
    
    def get_action_and_value(self, x, g = None, action=None):
        action, probs, ent = self.actor.get_action_dist(x,g,action)
        value = self.critic.get_value(x,g)
        return action,probs,ent,value
from typing import List
import numpy as np
import torch 
class Actor:
    def __init__(self, obs_space, act_space,goal_dim=0):
        self.obs_space = obs_space
        self.act_space = act_space
        self.goal_dim = goal_dim 
    def act(self,obs,**kwargs):
        return NotImplementedError
    def random_act(self, generator: np.random.Generator = None):
        if generator:
            return generator.uniform(self.act_space.low, self.act_space.high, size=self.act_space.low.size)
        else:
            return np.random.uniform(self.act_space.low, self.act_space.high, size=self.act_space.low.size)


class ZeroActor(Actor):
    """Output random actions."""
    def __init__(self, obs_space, act_space,goal_dim=0):
        """Init."""
        super().__init__(obs_space, act_space,goal_dim)

    def act(self, ob,**kwargs):
        """Act."""
        zero_act = self.act_space.low * 0
        return zero_act

class RandomActor(Actor):
    """Output random actions."""
    def __init__(self, obs_space, act_space,goal_dim=0, seed: int = 0):
        """Init."""
        super().__init__(obs_space, act_space,goal_dim)
        self.np_random = np.random.default_rng(seed=seed)

    def act(self, ob,**kwargs):
        """Act."""
        return self.random_act(generator=self.np_random)

class HeuristicActor(Actor):
    def __init__(self,obs_space,act_space,goal_dim=0,seed: int = 0):
        super().__init__(obs_space, act_space,goal_dim)
        self.np_random = np.random.default_rng(seed=seed)

    def act(self, ob,**kwargs):
        s = ob
        angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(
            s[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(s[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact

        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1., +1.)
        return a

class SafetyGymActorWrapper():
    def __init__(self,raw_actor):
        self.raw_actor = raw_actor
    def get_action(self,obs,deterministic,**kwargs):
        act, _, _, _ = self.raw_actor.step(
                    obs.float(), deterministic=deterministic
                )
        return act 


class ExpertActor(Actor):
    """output expert actions."""

    def __init__(self, obs_space, act_space, expert_agent,goal_dim = 0,device='cuda:0'):
        super().__init__(obs_space, act_space,goal_dim)
        self.agent = expert_agent
        self.device = device 
        
    def act(self, ob,**kwargs):
        """Act."""
        if not isinstance(ob,torch.Tensor):
            ob = torch.as_tensor(ob).to(self.device).reshape(1,-1)
        for k,v in kwargs.items():
            if not isinstance(v,torch.Tensor):
                kwargs[k] = torch.as_tensor(v).to(self.device).reshape(1,-1)
        act = self.agent.get_action(ob,deterministic=True,**kwargs)
        act = act.squeeze().detach().cpu().numpy()
        return act
    
class LaggyActor(Actor):
    """Laggy actor"""
    def __init__(self, obs_space, act_space, actor: Actor, repeat_prob: float, seed: int = 0,goal_dim = 0,):
        super().__init__(obs_space, act_space,goal_dim)
        self.actor = actor
        self.repeat_prob = repeat_prob
        self.actions = None
        self.np_random = np.random.default_rng(seed=seed)

        # TODO: Maze needs to maintain self.repeat (previously we repeated the action for 5 times)

    def act(self, ob, index=-1,**kwargs):
        """Act."""
        if index == -1:
            self.maybe_init_actions(num_envs=1)

        if self.np_random.random() < self.repeat_prob:
            return self.actions[index]
        else:
            self.actions[index] = self.actor.act(ob,**kwargs)
            return self.actions[index]

    def maybe_init_actions(self, num_envs):
        if self.actions is None:
            self.actions = [self.random_act(generator=self.np_random) for _ in range(num_envs)]

    # need to overwrite base class
    def batch_act(self, obss):
        self.maybe_init_actions(len(obss))
        actions = [self.act(obss[index], index) for index in range(len(obss))]
        return actions

class NoisedActor(Actor):
    """Noised Actor"""
    def __init__(self, obs_space, act_space, actor: Actor, noise_scale:float,
                 eps: float = 0, preserve_norm: bool = False, 
                 seed: int = 0,goal_dim = 0):
        super().__init__(obs_space, act_space,goal_dim)
        self.actor = actor
        self.eps = eps
        self._preserve_norm = preserve_norm
        self.noise_scale = noise_scale 
        self.np_random = np.random.default_rng(seed=seed)
    
    def _add_noise(self,action):
        noise = self.np_random.normal(size = action.shape) * self.noise_scale
        noised_action = action + noise 
        if self._preserve_norm:
            action_norm = np.linalg.norm(action,axis=-1)
            noised_action_norm = np.linalg.norm(noised_action,axis=-1)
            noised_action = noised_action / (noised_action + 1e-8) * action_norm
        return noised_action
    
    def act(self,ob,**kwargs):
        user_action = self.actor.act(ob,**kwargs)
        action = self._add_noise(user_action)
        # if self.np_random.random() < self.eps:
        #     action = self._add_noise(user_action)
        # else:
        #     action = user_action
        return action

            



class NoisyActor(Actor):
    """Noisy actor"""
    def __init__(self, obs_space, act_space, actor: Actor, 
                 eps: float, preserve_norm: bool = False, 
                 seed: int = 0,goal_dim = 0):
        super().__init__(obs_space, act_space,goal_dim)
        self.actor = actor
        self.eps = eps
        self._preserve_norm = preserve_norm
        self.repeat = 0
        self.np_random = np.random.default_rng(seed=seed)
        self.action = self.random_act(generator=self.np_random)

    def act(self, ob,**kwargs):
        """Act."""
        if self.repeat:
            self.repeat -= 1
            return self.action
        elif self.np_random.random() < self.eps:
        # elif np.random.rand() < self.eps:
            self.repeat = 0
            self.action = self.get_random(self.action)
            return self.action
        else:
            return self.actor.act(ob,**kwargs)

    def get_random(self, action):
        if self._preserve_norm:
            action0 = self.np_random.uniform(0.9, 1)
            action1 = self.np_random.uniform(0.9, 1)
            if self.np_random.random() < 0.5:
                action0 = -action0
            if self.np_random.random() < 0.5:
                action1 = -action1
            rand_action = np.array([action0, action1], dtype=action.dtype)

            """
            # theta = np.random.rand() * 2 * np.pi
            action0 = np.random.uniform(0.9, 1)
            action1 = np.random.uniform(0.9, 1)
            if np.random.rand() < 0.5:
                action0 = -action0
            if np.random.rand() < 0.5:
                action1 = -action1
            rand_action = np.array([action0, action1], dtype=action.dtype)
            # action = 0.0001 * np.array([np.cos(theta), np.sin(theta)], dtype=action.dtype)
            # print("use norm")
            # action = np.linalg.norm(action) * np.array([np.cos(theta), np.sin(theta)], dtype=action.dtype)
            """
            return rand_action
        else:
            rand_action = self.random_act(generator=self.np_random)
            # rand_action = self.act_space.sample()
            return rand_action
        

class SlowActor(Actor):
    """Slow actor"""
    def __init__(self, obs_space, act_space, actor: Actor, 
                 speed: float, preserve_norm: bool = False, 
                 seed: int = 0,goal_dim = 0):
        super().__init__(obs_space, act_space,goal_dim)
        self.actor = actor
        self.speed = speed

    def act(self, ob,**kwargs):
        """Act."""
        return (1-self.speed) * self.actor.act(ob,**kwargs)
from .base import Actor
import torch 
import numpy as np  

class DiffusionAssistedActor(Actor):
    def __init__(self,obs_space,act_space, diffusion_act, user_actor:Actor = None):
        super().__init__(obs_space,act_space)  
        self.diffusion_act = diffusion_act
        self.user_actor = user_actor 
    def act(self,obs, **kwargs):
        user_action = self.user_actor.act(obs,**kwargs)
        action = self.diffusion_act.act(obs,user_action)
        return action
    
    def get_user_action(self,obs,**kwargs):
        return self.user_actor.act(obs,**kwargs)

    def get_diffusion_action(self,obs,user_action):
        return self.diffusion_act.act(obs,user_action)
    
    def blend_act(self, obs, beta=0.5, **kwargs):
        user_action = self.user_actor.act(obs,**kwargs)
        action = self.diffusion_act.act(obs,user_action)
        # print("using blend action")
        blend_action = beta * user_action + (1 - beta) * action
        return blend_action
        
    
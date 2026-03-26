from .algo_utils import get_sigmas_karras

from .algo_utils import sample_deterministic_heun, sample_onestep,partial_onestep,partial_deterministic_heun,th_normalize_delta_obs
from .denoiser import KarrasDenoiser,JointConditionDenoiser,DDPMDenoiser
from diffusha.model.diffusion_model import JointConditionModel,ForwardModel
import numpy as np 
import torch 
from diffusha.algo.algo_utils import Transform


class EDMPolicy:
    def __init__(self,
                denoiser:KarrasDenoiser,
                model,
                steps,
                device='cpu',
                sigma_min=0.002,
                sigma_max=80,  
                rho=7.0,
                sampler_method="heun",
                s_churn=0.0,
                s_tmin=0.0,
                s_tmax=float("inf"),
                s_noise=1.0,
                 ) -> None:
        """
        Used for assisted actor. 
        Specify the parameters for the karras sampling and other post-preprocessing steps.
        """
        self.denoiser = denoiser
        self.model = model
        self.steps = steps
        self.sigmas = get_sigmas_karras(steps,sigma_min, sigma_max, rho, device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sampler_method = sampler_method
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.device = device

        self.sample_fn = {
        "deterministic_heun": sample_deterministic_heun,
                        }[sampler_method]
        if sampler_method in ["heun","deterministic_heun"]:
            self.sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
            )
        else:
            self.sampler_args = {}
    
    def _denoise(self,x, sigma, **model_kwargs):
        """
        Denoise the action using the denoiser. 
        """
        model_output, denoised = self.denoiser.denoise(self.model, x, sigma, **model_kwargs)
        return denoised
    
    
    def act(self,obs,user_action):
        ## We assume that all user_action is np.array 

        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs).to(self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
        if not isinstance(user_action, torch.Tensor):
            user_action = torch.as_tensor(user_action).to(self.device)
            if user_action.ndim == 1:
                user_action = user_action.unsqueeze(0)
        model_kwargs = {"condition": obs}
        action = self.sample_fn(
            self._denoise,
            user_action,
            self.sigmas,
            **self.sampler_args,
            **model_kwargs,
        )
        if isinstance(action, torch.Tensor):
            action = action.squeeze().cpu().numpy()
        return action 
        

class CMPolicy:
    def __init__(self,
                denoiser:KarrasDenoiser,
                model,
                transform:Transform,
                steps, ## maximum number of bins (discrete time steps)
                rho=7.0,
                sigma_min=0.002,
                sigma_max=80, 
                sampler_method="onestep",
                device='cpu',
                s_tmin=0.0,
                s_tmax=float("inf"),
                ts = None, ## For multi-step sampling
                 ) -> None:
        """
        Used for assisted actor. 
        Specify the parameters for the karras sampling and other post-preprocessing steps.
        """
        self.denoiser = denoiser
        self.model = model
        self.steps = steps
        self.sigmas = get_sigmas_karras(steps,sigma_min, sigma_max, rho, device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sampler_method = sampler_method
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.device = device
        self.transform = transform

        self.transform.min = torch.as_tensor(self.transform.min).to(self.device)
        self.transform.max = torch.as_tensor(self.transform.max).to(self.device)
        self.transform.mean = torch.as_tensor(self.transform.mean).to(self.device)
        self.transform.std = torch.as_tensor(self.transform.std).to(self.device)
        
        

        self.sample_fn = {
        "onestep": sample_onestep,
                        }[sampler_method]
        self.sampler_args = {}
    
    def _denoise(self,x, sigma, **model_kwargs):
        """
        Denoise the action using the denoiser. 
        """
        model_output, denoised = self.denoiser.denoise(self.model, x, sigma, **model_kwargs)
        return denoised
    
    
    def act(self,obs,user_action):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs).to(self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
        if not isinstance(user_action, torch.Tensor):
            user_action = torch.as_tensor(user_action).to(self.device)
            if user_action.ndim == 1:
                user_action = user_action.unsqueeze(0)

        ## Transform the user action
        if self.transform is not None:
            user_action = self.transform.transform(user_action) 

        model_kwargs = {"condition": obs}

        action = self.sample_fn(
            self._denoise,
            user_action,
            self.sigmas,
            **self.sampler_args,
            **model_kwargs,
        )
        ## Transform the user action
        if self.transform is not None:
            action = self.transform.inverse_transform(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze().cpu().numpy()
        
        return action 
    

class DDPMPolicy:
    def __init__(self,
                denoiser: DDPMDenoiser,
                fwd_diffuse_ratio: float,
                device = "cpu"):

        assert 0<=fwd_diffuse_ratio<=1 
        self.denoiser = denoiser
        self.fwd_diffuse_ratio = fwd_diffuse_ratio
        self.device = device
        self._k = int((self.denoiser.num_diffusion_steps - 1) * self.fwd_diffuse_ratio)
        print(f'forward diffusion steps for action: {self._k} / {self.denoiser.num_diffusion_steps}')
    
    def act(self,obs,user_action):
        ## We assume that all user_action is np.array 
        # TEMP: this is a temp solution for not landing. Do not apply to reacher
        # if (np.abs(user_action) < 0.1).all():
            # print("All action is 0 return user action.")
            # return user_action
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs).to(self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
        if not isinstance(user_action, torch.Tensor):
            user_action = torch.as_tensor(user_action).to(self.device)
            if user_action.ndim == 1:
                user_action = user_action.unsqueeze(0)
        model_kwargs = {"condition": obs}

        

        if torch.is_tensor(obs):
            cond = obs
            obs = torch.as_tensor(obs).to(self.device)
            user_action = torch.as_tensor(user_action).to(self.device)
            x_t = torch.cat((obs, user_action), axis=1)
        else:
            cond = torch.as_tensor(obs)
            x_t = torch.as_tensor(np.concatenate((obs, user_action), axis=1))
        
        cond_size = cond.shape[1]
        
        # forward diffuse 
        x_k, e = self.denoiser.diffuse(x_t, torch.as_tensor([self._k]))
        # reverse process
        x_0 = x_k
        # breakpoint()
        assert cond_size == self.denoiser.cond_dim, "cond_size does not match"
        x_0[..., :self.denoiser.cond_dim] = cond # only diffuse action 
        # LUZHE: This actually works
        for i in reversed(range(self._k)):
            x_0 = self.denoiser.p_sample(x_0, i)
            x_0[:, :self.denoiser.cond_dim] = cond  # Add condition
        
        # LUZHE: This is just a test
        # x_0, x_seq = self.denoiser.p_sample_loop(shape = x_0.shape,_k = self._k, start_x = x_0, cond = cond, naive_cond = False,)
        
        
        # this is assune x_0 is n*[[action+obs]] but it is possible that this is one dim then we need to remove the first part.
        action = x_0[..., self.denoiser.cond_dim: ] # user_action.shape[1]: action size

        return np.array(action.to("cpu")).squeeze()

                 
class JointConditionPolicy:
    def __init__(self,
                denoiser:JointConditionDenoiser,
                model:JointConditionModel,
                forward_model:ForwardModel,
                transform:Transform,
                steps, ## maximum number of bins (discrete time steps)
                noise_level,
                rho=7.0,
                sigma_min=0.002,
                sigma_max=80, 
                sampler_method="onestep",
                device='cuda:0',
                use_normalize_delta_obs = True,
                use_predict_next_state = False,
                 ) -> None:
        """
        Used for assisted actor. 
        Specify the parameters for the karras sampling and other post-preprocessing steps.
        """
        self.denoiser = denoiser
        self.model = model
        self.forward_model = forward_model
        self.steps = steps
        self.sigmas = get_sigmas_karras(steps,sigma_min, sigma_max, rho, device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sampler_method = sampler_method
        
        self.device = device
        self.transform = transform
        self.noise_level = noise_level 

        self.sample_fn = {
        "onestep": partial_onestep,
        'deterministic_heun': partial_deterministic_heun,
                        }[sampler_method]
        self.use_normalize_delta_obs = use_normalize_delta_obs
        self.use_predict_next_state = use_predict_next_state 
        self.list_norm = []
    
    def _denoise(self,x, sigma, **model_kwargs):
        """
        Denoise the action using the denoiser. 
        """
        model_output, denoised = self.denoiser.denoise(self.model, x, sigma, **model_kwargs)
        return denoised
    
    @torch.no_grad()
    def act(self,obs,user_action,deterministic=True):
        obs = torch.as_tensor(obs).to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        user_action = torch.as_tensor(user_action).to(self.device)
        if user_action.ndim == 1:
            user_action = user_action.unsqueeze(0)
        
        ## Use forward model to predict the next state
        # delta_direction, direction_norm, next_obs = self.forward_model.predict(obs,user_action,deterministic=deterministic)
        delta_direction, direction_norm, next_obs, average_norm = self.forward_model.predict(obs,user_action,deterministic=deterministic)

        ###### <DEBUG> ######
       
        ## Transform the user action
        user_action = self.transform.transform(user_action) 

        model_kwargs = {"condition1": obs,
                        "condition2":next_obs}
        if self.use_normalize_delta_obs:
            delta_obs = th_normalize_delta_obs(obs,next_obs)
            model_kwargs['condition2'] = delta_obs
        if not self.use_predict_next_state:
            model_kwargs['condition2'] = None
        
        action = self.sample_fn(
            self._denoise,
            user_action,
            self.sigmas,
            self.noise_level,
            **model_kwargs,
        )
        ## Transform the user action
        if self.transform is not None:
            action = self.transform.inverse_transform(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze().cpu().numpy()
        
        return action 

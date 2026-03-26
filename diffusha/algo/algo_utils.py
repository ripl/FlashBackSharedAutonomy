import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from copy import deepcopy
from abc import ABC, abstractmethod
import json 
import os 

def load_cm_models_path(dir):
    matched_paths = {}
    for root, dirs, files in os.walk(dir):
        for filename in files:
            # Match checkpoint filenames by suffix
            if '.cm_model.pth' in filename:
                file_path = os.path.join(root, filename)
                matched_paths['cm_model'] =file_path 
            elif '.forward_model.pth' in filename:
                file_path = os.path.join(root, filename)
                matched_paths['fwd_model'] =file_path 
            elif 'transform_stats.json' in filename:
                file_path = os.path.join(root, filename)
                matched_paths['transform_stats'] =root 
    return matched_paths




def decompose_state_dicts(state_dicts,is_grasp_idx):
    decomposed_list = []

    # Find the size of the first dimension (assuming all tensors have the same size in dim 0)
    num_slices = next(iter(next(iter(state_dicts.values())).values())).shape[0]

    # Create one dict per slice
    for i in range(num_slices):
        is_grasp = is_grasp_idx[i]
        if not is_grasp:
            continue
        sliced_dict = {}
        for k, v in state_dicts.items():
            sliced_dict[k] = {k_: v_[i].view(1, -1).detach().cpu() for k_, v_ in v.items()}
        decomposed_list.append(sliced_dict)

    return decomposed_list

def combined_cosine_polynomial_decay(init_ent_coef, final_ent_coef, iteration, num_iterations, power=2, frequency=5):
    """
    Combine cosine decay with polynomial decay for oscillating behavior that ends at zero.
    
    Args:
    - init_ent_coef: Initial entropy coefficient.
    - final_ent_coef: Final entropy coefficient (usually 0).
    - iteration: Current training iteration.
    - num_iterations: Total number of iterations.
    - power: The power for polynomial decay (controls steepness of the envelope).
    - frequency: Number of cosine oscillations over the iterations.

    Returns:
    - Entropy coefficient for the current iteration.
    """
    alpha = iteration / num_iterations
    # Polynomial envelope
    polynomial = (1 - alpha) ** power
    # Cosine wave (oscillates within [0, 1])
    cosine = 0.5 * (1 + np.cos(2 * np.pi * frequency * alpha))
    # Combine cosine wave with polynomial envelope
    return final_ent_coef + (init_ent_coef - final_ent_coef) * polynomial * cosine

def th_normalize_delta_obs(obs,obs2):
    delta_obs = obs2 - obs
    return delta_obs/(torch.norm(delta_obs,dim=-1,keepdim=True) + 1e-6)

class Transform:
    """
    Transform the input data to the range of [0,1] and then normalize it to the range of [-1,1] 
    """
    def __init__(self,mean,std,device = 'cuda:0') -> None:
        self.mean,self.std = mean,std 
        self.device = device 
        ## initialize the mean and std
        self.mean = torch.as_tensor(self.mean).to(self.device)
        self.std = torch.as_tensor(self.std).to(self.device)
        self.min = None 
        self.max = None 

        self.min_threshold = 1e-3
        self.max_threshold = 1e-3
        
    def fit_transform(self,x):

        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(self.device)
        
        self.min = torch.min(x,dim=0).values
        self.max = torch.max(x,dim=0).values
        ### One problem, due to the collection method, the min and max for some dimension may be all zeros!  so we could set a threshold to avoid all zero values  

        # Find the dimension that has min = max, then we set min = min - threshold, max = max + threshold
        min_max_equal = (self.max - self.min) < self.min_threshold
        self.min[min_max_equal] -= self.min_threshold/2.0
        self.max[min_max_equal] += self.max_threshold/2.0


        x = (x - self.min)/(self.max - self.min)  ## should be clipped in the range of [0,1]
        x = torch.clamp(x,0,1)
        x = (x - self.mean)/self.std
        x = torch.clamp(x,-1,1) 
        return x 

    def transform(self,x):
        x = (x - self.min)/(self.max - self.min)  ## should be clipped in the range of [0,1]
        x = torch.clamp(x,0,1)
        x = (x - self.mean)/(self.std)
        x = torch.clamp(x,-1,1)

        return x 
    def inverse_transform(self,x):
        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x).to(self.device)

        x = x * self.std + self.mean
        x = x * (self.max - self.min) + self.min
        return x

    def save_stats(self, file_dir):
        file_path = file_dir + '/transform_stats.json'
        stats = {
            'min': self.min.tolist() if self.min is not None else None,
            'max': self.max.tolist() if self.max is not None else None, 
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(stats, f)

    def load_stats(self, file_dir):
        file_path = file_dir + '/transform_stats.json'
        with open(file_path, 'r') as f:
            stats = json.load(f)
        self.min = torch.tensor(stats['min']).to(self.device)
        self.max = torch.tensor(stats['max']).to(self.device)
        self.mean = torch.tensor(stats['mean']).to(self.device)
        self.std = torch.tensor(stats['std']).to(self.device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def sample_multiple_bins( num_scales, batch_size):
        t = torch.randint(2,num_scales-2,[batch_size]).float()
        u = t - torch.ones_like(t)
        s = torch.cat(
            [torch.randint(0,int(u_i.item()), (1,)) for u_i in u]
        ).float()
       
        return t, u, s 



class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
    

class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights
    
    def sample_sigmas(self,batch_size, device):
        indices = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        t = self.diffusion.sigma_max ** (1 / self.diffusion.rho) + indices / (self.diffusion.num_timesteps - 1) * (
            self.diffusion.sigma_min ** (1 / self.diffusion.rho) - self.diffusion.sigma_max ** (1 / self.diffusion.rho)
        )
        t = t**self.diffusion.rho
        t = torch.clamp(t, min=self.diffusion.sigma_min, max=self.diffusion.sigma_max)
        weigths = torch.ones_like(t)
        return t, weigths
    
class LogNormalSampler():
    def __init__(self,diffusion, p_mean=-1.2, p_std=1.2):
        super().__init__()
        self.p_mean = p_mean
        self.p_std = p_std
        self.diffusion = diffusion

    def sample_sigmas(self, bs, device):
        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        weights = torch.ones_like(sigmas)
        sigmas = torch.clamp(sigmas, min=self.diffusion.sigma_min, max=self.diffusion.sigma_max)
        return sigmas, weights

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

@torch.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    **model_kwargs,
):
    """Single-step generation from a distilled model."""
    
    # print("Check One Step: ", sigmas[-10])
    s_in = x.new_ones([x.shape[0]])
    sigma = sigmas[-20]
    return distiller(x, sigma * s_in,**model_kwargs)

@torch.no_grad()
def partial_onestep(
    distiller,
    x,
    sigmas,
    level,
    **model_kwargs, 
    ):
    """
    Single-step generation from a distilled model.
    level: estimated level of the noise 
    sigmas: noise candidates
    """
    s_in = x.new_ones([x.shape[0]])
    sigma = sigmas[level]
    return distiller(x, sigma * s_in,**model_kwargs) 
@torch.no_grad()
def partial_deterministic_heun(
    denoiser,
    x,
    sigmas,
    level,
    **model_kwargs,
    ):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)[level:]

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in, **model_kwargs)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, **model_kwargs)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x



@torch.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    ts,
    t_min=0.002,
    t_max=40.0,
    rho=7.0,
    steps=80,
    **model_kwargs,
   
):
    # print("check: ", ts)
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in,**model_kwargs)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        
        x = x0 + torch.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x

@torch.no_grad()
def deterministic_iterative_sampler(
    distiller,
    x,
    ts,
    t_min=0.002,
    t_max=40.0,
    rho=7.0,
    steps=80,
    **model_kwargs,
  
):
    # print("check: ", ts)
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in,**model_kwargs)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 
    return x

@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    **model_kwargs 
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in, **model_kwargs)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x

@torch.no_grad()
def sample_deterministic_heun(
    denoiser,
    x,
    sigmas,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    **model_kwargs,
    
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    ## Rescale Action 
    x = x * 20 
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in, **model_kwargs)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma

        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, **model_kwargs)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x        

@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    **model_kwargs,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = denoiser(x, sigma_hat * s_in, **model_kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, **model_kwargs)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    ts=None,
    artifical_input = None 
):

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    x_T = torch.randn(*shape, device=device) * sigma_max
    if artifical_input is not None:
        x_T = artifical_input

    sample_fn = {
        "heun": sample_heun,
        "deterministic_heun": sample_deterministic_heun,
        "onestep": sample_onestep,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
        "deterministic_multistep": deterministic_iterative_sampler,
    }[sampler]

    if sampler in ["heun","deterministic_heun"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep" or sampler == "deterministic_multistep":
        sampler_args = dict(
             t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
        
        sigmas = ts
    else:
        sampler_args = {}

    def denoiser(x_t, sigma,model_kwargs):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        model_kwargs,
        **sampler_args,
    )
    return x_0


def cfg_sample(
    diffusion,
    model,
    shape,
    steps,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    ts=None,
    artifical_input = None,
    gamma = 1.0
):

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    x_T = torch.randn(*shape, device=device) * sigma_max
    if artifical_input is not None:
        x_T = artifical_input

    sample_fn = {
        "heun": sample_heun,
        "deterministic_heun": sample_deterministic_heun,
        "onestep": sample_onestep,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
        "deterministic_multistep": deterministic_iterative_sampler,
    }[sampler]

    if sampler in ["heun","deterministic_heun"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep" or sampler == "deterministic_multistep":
        sampler_args = dict(
             t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
        
        sigmas = ts
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if 'class_labels' in model_kwargs.keys():
            class_labels = model_kwargs['class_labels']
            zero_labels = torch.zeros_like(class_labels)
            zero_labels_kwargs = deepcopy(model_kwargs)
            zero_labels_kwargs['class_labels'] = zero_labels
            _, unc_denoised = diffusion.denoise(model, x_t, sigma, **zero_labels_kwargs)
            denoised = gamma * denoised + unc_denoised * (1-gamma)
        
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        **sampler_args,
    )
    return x_0

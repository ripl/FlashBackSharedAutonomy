import torch 
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from .algo_utils import append_dims, get_weightings, mean_flat, sample_multiple_bins

class JointConditionDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        num_timesteps = 80

    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.rho = rho
        self.num_timesteps = num_timesteps
    """ Utils """
    def get_snr(self, sigmas):
        return 1./(sigmas**2)

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    """ Compute Loss """
    
    def training_losses(self, model, x_start, sigmas, model_kwargs=None, gamma=0.0,weighting_model=None):
        """ 
        Simple version to compute the mixture loss for conditional and unconditional denoising loss
        when gamme = 1.0, it is the unconditional denoising loss
        when gamma = 0.0, it is the conditional denoising loss
        """
        if model_kwargs is None:
            model_kwargs = {'condition1':None,
                            'condition2':None}
        noise = torch.randn_like(x_start,device = x_start.device)
        terms = {}
        dims = x_start.ndim
        snrs = self.get_snr(sigmas)
        
        x_t = x_start + noise * append_dims(sigmas, dims)

        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)



        ## Unconditioned Output
        gamma = np.clip(gamma, 0.0, 1.0)
        if gamma > 0.0:
            ## clip the gamma in [0,1]
            unc_model_kwargs = {'condition1':model_kwargs['condition1'],
                                'condition2':None}
            model_output, unc_denoised = self.denoise(model, x_t, sigmas, **unc_model_kwargs)
            terms["xs_uc_mse"] = mean_flat((unc_denoised - x_start) ** 2)

        else:
            terms["xs_uc_mse"] = 0.0

        
        if weighting_model is not None:
            weights = weighting_model(sigmas)
            terms["mse"] = weights.exp()/x_start.shape[-1] * terms["xs_mse"] - weights
            terms["uc_mse"] = weights.exp()/x_start.shape[-1] * terms["xs_uc_mse"] - weights
        else:
            weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
            )
            terms["mse"] = weights * terms["xs_mse"]
            terms["uc_mse"] = weights * terms["xs_uc_mse"]
        
        ## Mixture Loss
        terms["loss"] = (1-gamma) * terms["mse"] + gamma * terms["uc_mse"]

        return terms
    
    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        weighting_model=None
    ):  
        """ 
        Currently, consistency loss only supports conditional CM 
        """

        if model_kwargs is None:
            model_kwargs = {'condition1':None,
                            'condition2':None}
                            
        noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]
        @torch.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )
        t = self.indice_to_time(indices, num_scales-1)
        t2 = self.indice_to_time(indices+1, num_scales-1)

        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = torch.get_rng_state()
        distiller = denoise_fn(x_t, t) 
        x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        diffs = mean_flat((distiller - distiller_target) ** 2)

        if weighting_model is not None:
            weights = weighting_model(t)
            loss = weights.exp()/x_start.shape[-1] * diffs - weights
        else:
            weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
            loss = weights * diffs
        terms = {}
        terms["loss"] = loss

        return terms
    

    def consistency_losses_v2(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        weighting_model=None,
        gamma = 0.5
    ):  
        """ 
        Currently, consistency loss only supports conditional CM 
        """

        if model_kwargs is None:
            model_kwargs = {'condition1':None,
                            'condition2':None}
                            
        noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t,model_kwargs):
            return self.denoise(model, x, t, **model_kwargs)[1]
        @torch.no_grad()
        def target_denoise_fn(x, t,model_kwargs):
            return self.denoise(target_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def teacher_denoise_fn(x, t,model_kwargs):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def heun_solver(samples, t, next_t, x0,model_kwargs):
            x = samples
            denoiser = teacher_denoise_fn(x, t,model_kwargs)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            denoiser = teacher_denoise_fn(samples, next_t,model_kwargs)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )
        t = self.indice_to_time(indices, num_scales-1)
        t2 = self.indice_to_time(indices+1, num_scales-1)
        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = torch.get_rng_state()

        ### Conditional Denoising ###
        distiller = denoise_fn(x_t, t,model_kwargs) 
        x_t2 = heun_solver(x_t, t, t2, x_start,model_kwargs).detach()

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2,model_kwargs)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        diffs = mean_flat((distiller - distiller_target) ** 2)

        ### Unconditional Denoising ### 
        unc_model_kwargs = {'condition1':model_kwargs['condition1'],
                            'condition2':None}
        distiller_uc = denoise_fn(x_t, t,unc_model_kwargs)
        x_t2_uc = heun_solver(x_t, t, t2, x_start,unc_model_kwargs).detach()
        distiller_target_uc = target_denoise_fn(x_t2_uc, t2,unc_model_kwargs)
        distiller_target_uc = distiller_target_uc.detach()
        diffs_uc = mean_flat((distiller_uc - distiller_target_uc) ** 2)
        

        if weighting_model is not None:
            weights = weighting_model(t)
            loss = weights.exp()/x_start.shape[-1] * diffs - weights
            uc_loss = weights.exp()/x_start.shape[-1] * diffs_uc - weights
        else:
            weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
            loss = weights * diffs
            uc_loss = weights * diffs_uc
        terms = {}
        terms["loss"] = (1-gamma) * loss + gamma * uc_loss
        terms['c_loss'] = loss
        terms['uc_loss'] = uc_loss

        return terms
    

    def indice_to_time(self, indices, num_scales):
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho
        return t
    
    """ Denoise"""
    def denoise(self, model, x_t, sigmas, **model_kwargs):
        import torch.distributed as dist

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44) ## rescale ? 
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l2",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.rho = rho
        self.num_timesteps = 80
    """ Utils """
    def get_snr(self, sigmas):
        return 1./(sigmas**2)

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    """ Compute Loss """
    
    def training_losses(self, model, x_start, sigmas, model_kwargs=None, gamma=0.0):
        """ Gamma is label dropout rate"""
        if model_kwargs is None:
            model_kwargs = {'condition':None}
        noise = torch.randn_like(x_start,device = x_start.device)
        terms = {}
        dims = x_start.ndim
        # print("Check: ", x_start.shape, noise.shape, sigmas.shape)
        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        x_t = x_start + noise * append_dims(sigmas, dims)

        ## Conditioned Output
        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        ## Unconditioned Output
        if gamma > 0.0:
            unc_model_kwargs = {'condition':None}
            model_output, unc_denoised = self.denoise(model, x_t, sigmas, **unc_model_kwargs)
            terms["xs_uc_mse"] = mean_flat((unc_denoised - x_start) ** 2)
            terms["uc_mse"] = mean_flat(weights * (unc_denoised - x_start) ** 2)
            terms["loss"] = terms["mse"] + gamma * terms["uc_mse"]
        else:
            terms["loss"] = terms["mse"]

        return terms
    
    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
    ):  
        
        """Try: Mask the label at the beginning and end of the diffusion process, keep gamma% of the label"""

        if model_kwargs is None:
            model_kwargs = {'condition':None}
        noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        if target_model:

            @torch.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_model, x, t, **model_kwargs)[1]

        else:
            raise NotImplementedError("Must have a target model")

        if teacher_model:

            @torch.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @torch.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = torch.get_rng_state()

        distiller = denoise_fn(x_t, t)  # distilled one-step output (end-to-end from x_t) 

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)


        if self.loss_norm == "l1":
            diffs = torch.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms
    
    def indice_to_time(self, indices, num_scales):
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho
        return t
    
    """ Denoise"""
    def denoise(self, model, x_t, sigmas, **model_kwargs):
        import torch.distributed as dist

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44) ## rescale ? 

        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

import random
from diffusha.algo.ddpm_base import make_beta_schedule, extract, ConditionalModel

class DDPMDenoiser:
    def __init__(
            self,
            model: ConditionalModel,
            num_diffusions_steps: int,
            beta_schedule: str,
            beta_min: float,
            beta_max: float,
            cond_dim: int = 0,
            seed: int = 234,
            device = "cuda:0"
    ) -> None:
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # configuration
        self.cond_dim = cond_dim
        self.device = device
        
        # Initialize beta schedule and diffusion parameters
        betas = make_beta_schedule(
            schedule=beta_schedule,
            n_timesteps=num_diffusions_steps,
            start=beta_min,
            end=beta_max
        )

        self.betas = betas.to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1-self.alphas_prod)

        self.model = model
        self.num_diffusion_steps = num_diffusions_steps
        self.predict_epsilon = True

        self.model.to(self.device)


    @torch.no_grad()
    def diffuse(
            self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        assert len(x_0.shape) > 1, "x_0 must have batch dimension"
        assert len(t.shape) == 1, f"Wrong shape for t: {t.shape}"
        
        x_0 = x_0.to(self.device)
        t = t.to(self.device)

        a = extract(self.alphas_bar_sqrt, t, x_0)
        am1 = extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        e = torch.rand_like(x_0)

        x_t = x_0 * a + e * am1

        # # overwrite condition to keep condition not change
        if self.cond_dim > 0:
            x_t[..., :self.cond_dim ] = x_0[..., :self.cond_dim]
        #     # x_t[..., : self.cond_dim] = x_0[..., :self.cond_dim]

        return x_t, e
    

    def noise_estimate_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """calculate the noise esitimation loss"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_diffusion_steps, size = (batch_size // 2 +1,))
        t = torch.cat([t, self.num_diffusion_steps - t - 1], dim=0)[:batch_size].long()
        t = t.to(self.device)

        x_t, e = self.diffuse(x_0, t)
        x_t = x_t.float()
        t = t.long()
        output = self.model(x_t, t)

        if self.predict_epsilon:
            err = (e - output)
        else:
            err = (x_0 - output)

        return err.square().mean()
    

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """ reverse diffusion process """
        x = x.float().to(self.device)
        t_tensor = torch.tensor([t]).to(self.device)

        eps_factor = (1 - extract(self.alphas, t_tensor, x)) / extract(self.one_minus_alphas_bar_sqrt, t_tensor, x)

        # breakpoint()

        if self.predict_epsilon:
            eps_theta = self.model(x, t_tensor)
            mean = (1 / extract(self.alphas, t_tensor, x).sqrt()) * (
                x - eps_factor * eps_theta
            )
        else:
            mean = self.model(x, t_tensor)
        
        z = torch.randn_like(x)
        sigma_t = extract(self.betas, t_tensor, x).sqrt()
        return mean + sigma_t * z

    
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        _k: int = None,
        start_x: torch.Tensor = None,
        cond: torch.Tensor = None,
        naive_cond: bool = False,
    ):
        """Peforms conditional sampling (if cond is not None)

        This assumes cond Tensor corresponds to the first cond.shape[0] dimension of diffusion state space.
        """

        def apply_naive_condition(x: torch.Tensor, cond: torch.Tensor, timestep: int):
            """Simply replace a part of x with cond"""
            assert len(cond.shape) in [1, 2], f"Wrong shape on cond: {cond.shape}"
            assert len(cond.shape) == len(
                x.shape
            ), "Condition shape does not match sample shape"
            cond_dim = cond.shape[-1]
            x[..., :cond_dim] = cond
            return x

        def apply_condition(x: torch.Tensor, cond: torch.Tensor, timestep: int):
            """A better way: Replace a part of x with noisy cond tensor"""
            assert len(cond.shape) in [1, 2], f"Wrong shape on cond: {cond.shape}"
            assert len(cond.shape) == len(
                x.shape
            ), f"Condition shape does not match sample shape"
            cond_dim = cond.shape[-1]
            assert cond_dim == self.cond_dim, f"Condition dim does not match model cond dim: {cond_dim} != {self.cond_dim}"

            if len(cond.shape) == 1:
                naive_x = torch.zeros_like(x).unsqueeze(0)  # Add batch dim
            else:
                naive_x = torch.zeros_like(x)  # Add batch dim

            naive_x[
                ..., :cond_dim
            ] = cond  # TODO: The values of the rest of dimensions shouldn't matter, but should double check as Luzhe mentionied
            naive_x = naive_x.to(self.device)
            k = torch.as_tensor([timestep]).to(self.device)
            noisy_x, _ = self.diffuse(naive_x, k)
            noisy_x = noisy_x.squeeze(0)  # Remove batch dim
            _cond = noisy_x[..., :cond_dim]

            x[..., :cond_dim] = _cond
            return x

        _apply_cond = apply_naive_condition if naive_cond else apply_condition

        # Use start_x if specified
        if start_x is not None:
            assert shape == start_x.shape
            x = start_x
        else:
            x = torch.randn(shape)

        x_seq = []
        for k in reversed(range(_k if _k is not None else self.num_diffusions_steps)):
            if cond is not None:
                x = _apply_cond(x, cond, k)
            x_seq.append(x.detach().cpu())
            x = self.p_sample(x, k)

        # Don't forget to append the last one
        x_seq.append(x.detach().cpu())

        return x, x_seq
    
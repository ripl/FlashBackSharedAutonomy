import torch 
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .algo_utils import update_ema, UniformSampler,LogNormalSampler,Transform,th_normalize_delta_obs
import wandb
import time
from datetime import datetime
import os 
from diffusha.model.model_utils import load_model
from diffusha.data_collection.buffer import ExpertTransitionDataset_v2
from diffusha.model.diffusion_model import JointConditionModel
from diffusha.algo.denoiser import JointConditionDenoiser
from torch.utils.tensorboard import SummaryWriter
from diffusha.algo.expert_trainer import ForwardTrainer



class JointEDMTrainer:
    def __init__(self, 
                 model:JointConditionModel,
                 diffusion:JointConditionDenoiser,
                 weighting_model,
                 args) -> None:
        self.args = args 
        self.model=model.to(self.args.device)
        self.ema_model = deepcopy(model).requires_grad_(False)
        self.diffusion = diffusion 
        self.scheduler_sampler = LogNormalSampler(self.diffusion,p_mean=-1.2,p_std=1.6)
        self.lr = self.args.lr
        self.ema_rate = self.args.ema_rate
        self.weighting_model = weighting_model
        if self.weighting_model is not None:
            self.weighting_model = weighting_model.to(self.args.device)
            self.optim = torch.optim.RAdam(list(self.model.parameters())+list(self.weighting_model.parameters()), lr = self.lr)
        else:
            self.optim = torch.optim.RAdam(self.model.parameters(), lr = self.lr)
        self.device = self.args.device  
        self.transform = Transform(mean = 0.5, std = 0.5, device = self.device)
    
    def save(self):
        # Get current date and time
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")  # Folder named by the date
        timestamp_str = current_time.strftime("%H-%M-%S")  # Timestamp for the file names
        # Create a directory for the current day if it doesn't exist
        save_dir = os.path.join(self.args.save_dir,self.args.exp_name, date_str)
        os.makedirs(save_dir, exist_ok=True)

        # Define file paths for actor, qf1, and qf2 within the same subfolder
        model_save_path = os.path.join(save_dir, f"{self.args.exp_name}_{timestamp_str}_{self.args.seed}.edm_model.pth")
        ema_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}_{timestamp_str}_{self.args.seed}.ema_edm_model.pth")

        torch.save(self.model.state_dict(), model_save_path)
        torch.save(self.ema_model.state_dict(), ema_model_save_path)

        # Optionally, print the paths to confirm
        print(f"EDM model saved at: {model_save_path}")
        print(f"ema_EDM model saved at: {ema_model_save_path}")

        ## also save the transform 
        transform_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.edm_transform.pkl")
        torch.save(self.transform,transform_path)
        self.transform.save_stats(save_dir)
        print(f"Transform saved at: {transform_path}")

        ## if weighting model is not None, save the weighting model
        if self.weighting_model is not None:
            weighting_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.edm_weighting_model.pth")
            torch.save(self.weighting_model.state_dict(), weighting_model_save_path)
            print(f"Weighting model saved at: {weighting_model_save_path}")


    def run_loop(self, n_iter, dataset:ExpertTransitionDataset_v2, batch_size, gamma=1.0,forward_trainer:ForwardTrainer = None):
        run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        # Initialize logging
        if self.args.track:
            wandb.init(
                project=self.args.wandb_project_name,
                dir = self.args.wandb_logdir,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                name=run_name,
                save_code=True,
            )
        log_dir = os.path.join(self.args.log_dir,run_name)
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        ## fit the stats for the action space
        self.transform.fit_transform(dataset.act)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        start_time = time.time()
        
        for i in tqdm(range(n_iter), position=0, leave=True):
            uc_loss_queue = []
            c_loss_queue = []
            forward_loss_queue = []            
            for obs,act,obs2 in dataloader:  # obs, act, obs2
                self.optim.zero_grad()
                obs = obs.to(self.device)
                act = act.to(self.device)
                obs2 = obs2.to(self.device)
                conds = {'condition1': obs, 'condition2': obs2}
                if self.args.use_normalize_delta_obs:
                    normalized_delta_obs = th_normalize_delta_obs(obs,obs2)
                    conds['condition2'] = normalized_delta_obs
                
                normalized_act = self.transform.transform(act)
                t, _ = self.scheduler_sampler.sample_sigmas(obs.shape[0], device=obs.device)
                losses = self.diffusion.training_losses(self.model,normalized_act,t, model_kwargs=conds, gamma=gamma,weighting_model=self.weighting_model)
                loss = losses["loss"].mean()

                loss.backward()

                self.optim.step()
                update_ema(self.ema_model.parameters(), self.model.parameters(), rate=self.ema_rate)
                c_loss_queue.append(losses["mse"].mean().item())
                uc_loss_queue.append(losses["uc_mse"].mean().item())

                ## if forward trainer is not None, update the forward trainer
                if forward_trainer is not None:
                    forward_loss,forward_loss_info = forward_trainer.update_onestep(obs,act,obs2)
                    forward_loss_queue.append(forward_loss.item())
            
            # Log metrics every 100 iterations
            if i % 100 == 0:
                c_avg_loss = np.mean(c_loss_queue)
                uc_avg_loss = np.mean(uc_loss_queue)
                # Logging loss and step to TensorBoard
                writer.add_scalar("losses/c_train_loss", c_avg_loss, i)
                writer.add_scalar("losses/uc_train_loss", uc_avg_loss, i)
                if forward_trainer is not None:
                    avg_forward_loss = np.mean(forward_loss_queue)
                    writer.add_scalar("losses/forward_train_loss", avg_forward_loss, i)
                    for k,v in forward_loss_info.items():
                        writer.add_scalar(f"forward_loss/{k}",v,i)
                
                c_loss_queue = []  
                uc_loss_queue = [] 
                forward_loss_queue = []  # Reset the forward loss queue

            # Log SPS (Steps per Second) every 100 iterations
            if i % 100 == 0:
                sps = i / (time.time() - start_time)
                writer.add_scalar("charts/SPS", sps, i)
                # if self.args.track:
                #     wandb.log({"SPS": sps})

            if (i+1)%self.args.save_every == 0:
                self.save()
        writer.close()
        if self.args.track:
            wandb.finish()


class JointCMTrainer:
    def __init__(self, 
                 model:JointConditionModel,
                 diffusion:JointConditionDenoiser,
                 target_model:JointConditionModel,
                 teacher_model:JointConditionModel,
                 teacher_diffusion:JointConditionDenoiser,
                 weighting_model,
                 args) -> None:
        self.args = args

        self.ema_rate = self.args.ema_rate 
        self.target_rate = self.args.target_rate
        self.device = self.args.device
        self.lr = self.args.lr

        # Init Student Model 
        self.model=model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.ema_model = deepcopy(model).requires_grad_(False)

        self.diffusion = diffusion 

        # Init Teacher Model & Load Model 
        self.teacher_model = teacher_model.to(self.device)
        if self.args.load_teacher_model:
            self.teacher_model = load_model(self.teacher_model,self.args.teacher_model_path,eval = True,device=self.device)
        self.teacher_diffusion = teacher_diffusion
        

        self.target_model.requires_grad_(False)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.scheduler_sampler = UniformSampler(self.diffusion)
        self.weighting_model = weighting_model
        if self.weighting_model is not None:
            self.weighting_model = weighting_model.to(self.device)
            self.optim = torch.optim.RAdam(list(self.model.parameters())+list(self.weighting_model.parameters()), lr = self.lr)
        else:   
            self.optim = torch.optim.RAdam(self.model.parameters(), lr = self.lr)

        self.transform = Transform(mean = 0.5, std = 0.5, device = self.device)
    
    def target_rate_decay(self, initial_target_rate, decay_T, t):
        return initial_target_rate * np.exp(-t / decay_T)
    
    def save(self):
        # Get current date and time
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")  # Folder named by the date
        timestamp_str = current_time.strftime("%H-%M-%S")  # Timestamp for the file names
        # Create a directory for the current day if it doesn't exist
        save_dir = os.path.join(self.args.save_dir, self.args.exp_name, date_str)
        os.makedirs(save_dir, exist_ok=True)

        # Define file paths for actor, qf1, and qf2 within the same subfolder
        model_save_path = os.path.join(save_dir, f"{self.args.exp_name}_{timestamp_str}_{self.args.seed}.cm_model.pth")
        ema_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}_{timestamp_str}_{self.args.seed}.ema_cm_model.pth")

        torch.save(self.model.state_dict(), model_save_path)
        torch.save(self.ema_model.state_dict(), ema_model_save_path)

        # Optionally, print the paths to confirm
        print(f"CM model saved at: {model_save_path}")
        print(f"ema_CM model saved at: {ema_model_save_path}")

        ## also save the transform 
        transform_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.cm_transform.pkl")
        torch.save(self.transform,transform_path)
        self.transform.save_stats(save_dir)
        print(f"Transform saved at: {transform_path}")

        ## if weighting model is not None, save the weighting model
        if self.weighting_model is not None:
            weighting_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.cm_weighting_model.pth")
            torch.save(self.weighting_model.state_dict(), weighting_model_save_path)
            print(f"Weighting model saved at: {weighting_model_save_path}")
        if self.forward_trainer is not None:
            self.forward_trainer.save()            
            print(f"Forward trainer saved")
        
    
    def run_loop(self, n_iter, dataset:ExpertTransitionDataset_v2, batch_size,gamma=1.0,forward_trainer:ForwardTrainer = None):
        run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.forward_trainer = forward_trainer
        self.save()

        # Initialize logging
        if self.args.track:
            wandb.init(
                project=self.args.wandb_project_name,
                dir = self.args.wandb_logdir,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                name=run_name,
                save_code=True,
            )
        log_dir = os.path.join(self.args.log_dir,run_name)
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
        dataset.act =  self.transform.fit_transform(dataset.act)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        start_time = time.time()
        # for i in tqdm(range(n_iter),position=0, leave=True):
        propress_bar = tqdm(range(n_iter),position=0, leave=True)
        for i in propress_bar:
            uc_loss_queue = []
            c_loss_queue = []
            forward_loss_queue = []
            for obs,act,obs2 in dataloader:  # obs, act, obs2
                self.optim.zero_grad()
                obs = obs.to(self.device)
                act = act.to(self.device)
                obs2 = obs2.to(self.device)
                conds = {'condition1': obs, 'condition2': obs2}
                normalized_act = self.transform.transform(act)

                normalized_delta_obs = th_normalize_delta_obs(obs,obs2)
                if self.args.use_normalize_delta_obs:
                    conds['condition2'] = normalized_delta_obs
                
                gamma = np.clip(gamma, 0.0, 1.0)

                lossess = self.diffusion.consistency_losses_v2(
                    model = self.model,
                    x_start=normalized_act,
                    num_scales=self.args.discrete_steps,
                    model_kwargs=conds,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    weighting_model=self.weighting_model,
                    gamma = gamma
                )
                loss = lossess["loss"].mean()
                c_loss = lossess["c_loss"].mean()
                uc_loss = lossess["uc_loss"].mean()

                loss.backward()
                self.optim.step()

                ## Update Model, Moving Update Target Model 
                update_ema(self.ema_model.parameters(), self.model.parameters(), rate=self.ema_rate)
                self._update_target_ema(self.target_rate_decay(self.target_rate, n_iter, i))
                
                c_loss_queue.append(c_loss.item())  
                uc_loss_queue.append(uc_loss.item())

                ## if forward trainer is not None, update the forward trainer
                if forward_trainer is not None:
                    forward_loss,forward_loss_info = forward_trainer.update_onestep(obs,act,obs2)
                    forward_loss_queue.append(forward_loss.item())

            # Log metrics every 100 iterations
            if i % 100 == 0:
                c_avg_loss = np.mean(c_loss_queue)
                uc_avg_loss = np.mean(uc_loss_queue)
                # Logging loss and step to TensorBoard
                writer.add_scalar("losses/c_train_loss", c_avg_loss, i)
                writer.add_scalar("losses/uc_train_loss", uc_avg_loss, i)
                if forward_trainer is not None:
                    avg_forward_loss = np.mean(forward_loss_queue)
                    writer.add_scalar("losses/forward_train_loss", avg_forward_loss, i)
                    for k,v in forward_loss_info.items():
                        writer.add_scalar(f"forward_loss/{k}",v,i)
                c_loss_queue = []  
                uc_loss_queue = [] 
                forward_loss_queue = []  # Reset the forward loss queue

            # Log SPS (Steps per Second) every 100 iterations
            if i % 100 == 0:
                sps = i / (time.time() - start_time)
                writer.add_scalar("charts/SPS", sps, i)

            if (i+1)%self.args.save_every == 0:
                self.save()
        writer.close()
        if self.args.track:
            wandb.finish()

    def _update_target_ema(self,update_rate):
        with torch.no_grad():
            update_ema(
                self.target_model.parameters(),
                self.model.parameters(),
                rate=update_rate,
            )
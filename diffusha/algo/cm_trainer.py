import torch 
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .algo_utils import update_ema, UniformSampler
import wandb
import time
from datetime import datetime
import os 
from diffusha.model.model_utils import load_model
from .algo_utils import Transform
from diffusha.data_collection.buffer import ExpertTransitionDataset_v2
from torch.utils.tensorboard import SummaryWriter

class CMTrainer:
    def __init__(self, 
                 model,
                 diffusion,
                 target_model,
                 teacher_model,
                 teacher_diffusion,
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
        self.teacher_diffusion = load_model(self.teacher_model,self.args.teacher_model_path,eval = True,device=self.device)
        self.teacher_diffusion = teacher_diffusion
        

        self.target_model.requires_grad_(False)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.scheduler_sampler = UniformSampler(self.diffusion)
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
        save_dir = os.path.join(self.args.save_dir,self.args.exp_name, date_str, timestamp_str)
        os.makedirs(save_dir, exist_ok=True)

        # Define file paths for actor, qf1, and qf2 within the same subfolder
        model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.model.pth")
        target_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.target_model.pth")
        ema_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.ema_model.pth")
        teacher_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.teacher_model.pth")

        torch.save(self.model.state_dict(), model_save_path)
        torch.save(self.target_model.state_dict(),target_model_save_path)
        torch.save(self.ema_model.state_dict(), ema_model_save_path)
        torch.save(self.teacher_model.state_dict(), teacher_model_save_path)

        # Optionally, print the paths to confirm
        print(f"CM model saved at: {model_save_path}")
        print(f"ema_CM model saved at: {ema_model_save_path}")

        # also save the transform
        transform_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.transform.pkl")
        torch.save(self.transform,transform_path)
        print(f"Transform saved at: {transform_path}")
        
    
    def run_loop(self, n_iter, dataset:ExpertTransitionDataset_v2, batch_size):
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
        dataset.act =  self.transform.fit_transform(dataset.act)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        start_time = time.time()
        # for i in tqdm(range(n_iter),position=0, leave=True):
        for i in range(n_iter):
            loss_queue = []
            for y,x in dataloader: 
                self.optim.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                conds = {'condition': y}

                losses = self.diffusion.consistency_losses(
                    model = self.model,
                    x_start=x,
                    num_scales=self.args.discrete_steps,
                    model_kwargs=conds,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                )

                loss = (losses["loss"]).mean()
                loss.backward()
                self.optim.step()

                ## Update Model, Moving Update Target Model 
                update_ema(self.ema_model.parameters(), self.model.parameters(), rate=self.ema_rate)
                self._update_target_ema(self.target_rate_decay(self.target_rate, n_iter, i))
                loss_queue.append(loss.item())

            # Log metrics every 100 iterations
            if i % 100 == 0:
                avg_loss = np.mean(loss_queue)
                # Logging loss and step to TensorBoard
                writer.add_scalar("losses/train_loss", avg_loss, i)
                # Optional: Log to wandb if tracking is enabled
                if self.args.track:
                    wandb.log({"train_loss": avg_loss, "step": i})
                loss_queue = []  # Reset the loss queue

            # Log SPS (Steps per Second) every 100 iterations
            if i % 100 == 0:
                sps = i / (time.time() - start_time)
                writer.add_scalar("charts/SPS", sps, i)
                if self.args.track:
                    wandb.log({"SPS": sps})

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
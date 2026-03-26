import torch 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb 
import time
import os 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from diffusha.algo.ddpm_base import EMA
import torch.optim as optim
from diffusha.algo.denoiser import DDPMDenoiser

class DDPMTrainer:
    def __init__(self, 
                 diffusion: DDPMDenoiser,
                 args) -> None:
        """Should expect args have: obs_size, act_size, save_every, eval_every"""
        self.diffusion = diffusion

        self.args = args 
        
        self.device = self.args.device
        self.diffusion = diffusion 
        self.lr = self.args.lr
        self.ema_rate = self.args.ema_rate
        self.ema = EMA(self.ema_rate)
        self.ema.register(self.diffusion.model)
        self.optim = optim.Adam(self.diffusion.model.parameters(), lr = self.lr)
    
    
    def train_step(self, batch: torch.Tensor) -> float:
        loss = self.diffusion.noise_estimate_loss(batch)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), 1.0)
        self.optim.step()
        self.ema.update(self.diffusion.model)
        return loss
    
    
    def save(self):
        # Get current date and time
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")  # Folder named by the date
        timestamp_str = current_time.strftime("%H-%M-%S")  # Timestamp for the file names
        # Create a directory for the current day if it doesn't exist
        save_dir = os.path.join(self.args.save_dir, self.args.exp_name, date_str)
        os.makedirs(save_dir, exist_ok=True)

        # Define file paths for actor, qf1, and qf2 within the same subfolder
        model_save_path = os.path.join(save_dir, f"{self.args.exp_name}_{timestamp_str}_{self.args.seed}.model.pth")
        ema_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}_{timestamp_str}_{self.args.seed}.ema_model.pth")

        torch.save(self.diffusion.model.state_dict(), model_save_path)
        torch.save(self.ema.state_dict(), ema_model_save_path)

        # Optionally, print the paths to confirm
        print(f"DDPM model saved at: {model_save_path}")
        print(f"ema_DDPM model saved at: {ema_model_save_path}")


    def run_loop(self, n_iter, dataset, batch_size):
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
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        start_time = time.time()

        for i in tqdm(range(n_iter), position=0, leave=True):
            loss_queue = []
            
            for y, x in dataloader:  # obs, act
                # breakpoint()
                x = x.to(self.device)
                y = y.to(self.device)
                x_0 = torch.concatenate((y, x), dim=1)

                loss = self.train_step(x_0)
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
    
    
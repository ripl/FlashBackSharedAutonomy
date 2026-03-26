import torch 
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from .algo_utils import update_ema,LogNormalSampler
import wandb 
import time
import os 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .algo_utils import Transform
from diffusha.data_collection.buffer import ExpertTransitionDataset_v2
class EDMTrainer:
    def __init__(self, 
                 model,
                 diffusion,
                 args) -> None:
        self.args = args 
        self.model=model.to(self.args.device)
        self.ema_model = deepcopy(model).requires_grad_(False)
        self.diffusion = diffusion 
        # self.scheduler_sampler = UniformSampler(self.diffusion)
        self.scheduler_sampler = LogNormalSampler(self.diffusion,p_mean=-1.2,p_std=1.6)
        self.lr = self.args.lr
        self.ema_rate = self.args.ema_rate
        self.optim = torch.optim.RAdam(self.model.parameters(), lr = self.lr)
        self.device = self.args.device  
        self.transform = Transform(mean = 0.5, std = 0.5, device = self.device)
    
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
        ema_model_save_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.ema_model.pth")

        torch.save(self.model.state_dict(), model_save_path)
        torch.save(self.ema_model.state_dict(), ema_model_save_path)

        # Optionally, print the paths to confirm
        print(f"EDM model saved at: {model_save_path}")
        print(f"ema_EDM model saved at: {ema_model_save_path}")

        ## also save the transform 
        transform_path = os.path.join(save_dir, f"{self.args.exp_name}__{self.args.seed}.transform.pkl")
        torch.save(self.transform,transform_path)
        print(f"Transform saved at: {transform_path}")


    def run_loop(self, n_iter, dataset:ExpertTransitionDataset_v2, batch_size, gamma=1.0):
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

        for i in tqdm(range(n_iter), position=0, leave=True):
            loss_queue = []
            
            for y, x in dataloader:  # obs, act
                self.optim.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                conds = {'condition': y}
                t, weights = self.scheduler_sampler.sample_sigmas(x.shape[0], device=x.device)

                losses = self.diffusion.training_losses(self.model, x, t, model_kwargs=conds, gamma=gamma)
                loss = (losses["loss"] * weights).mean()
                loss.backward()
                self.optim.step()
                update_ema(self.ema_model.parameters(), self.model.parameters(), rate=self.ema_rate)
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
    
    
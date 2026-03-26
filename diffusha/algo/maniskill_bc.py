import torch 
from ..model.ppo import ActorCritic_v3 as ActorCritic
from ..model.diffusion_model import ForwardModel
from tqdm import tqdm
import time 
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from diffusha.data_collection.buffer import RolloutBuffer
from torch.utils.data import DataLoader,Dataset
import os
from datetime import datetime
from collections import deque,defaultdict
import torch.nn as nn 
from diffusha.algo.algo_utils import combined_cosine_polynomial_decay
from pathlib import Path

class BCDataset(Dataset):
    def __init__(self,dir, state_dim, goal_dim, act_dim, add_noise = False):
        super().__init__()
        self.dir = Path(dir)
        self.state_dim = state_dim
        self.goal_dim = goal_dim 
        self.act_dim = act_dim 
        self.add_noise = add_noise
        self._file_cache = []
        for fname in self.dir.iterdir():
            buff = torch.load(fname)
            if not isinstance(buff, torch.Tensor):
                buff = torch.tensor(buff).to()
            self._file_cache.append(buff)
        self.total_file = torch.cat(self._file_cache, axis=0)
        self.obs = self.total_file[:,0:self.state_dim]
        self.act = self.total_file[:,self.state_dim:self.state_dim+self.act_dim]
        self.obs2 = self.total_file[:,self.state_dim+self.act_dim:2*self.state_dim+self.act_dim]
        self.goal = self.total_file[:,2*self.state_dim+self.act_dim : 2*self.state_dim+self.act_dim + self.goal_dim]
        ## keep some statistics for noise 
        self.obs_std = torch.std(self.obs, dim=0)
        self.obs_noise_vec = self.obs_std * 0.1
        obs_min_noise = torch.ones(self.state_dim) * 1e-4
        self.obs_noise_vec = torch.max(self.obs_noise_vec, obs_min_noise) 
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        # Return data and label at the specified inde
        obs = self.obs[idx]
        act = self.act[idx]
        goal = self.goal[idx]
        if self.add_noise:
            obs = obs + torch.randn_like(obs) * self.obs_noise_vec
        return obs,act,goal

class BCTrainer:
    def __init__(self, env,args):
        self.args = args
      
        device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.device = device
        goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
        self.num_envs = env.unwrapped.num_envs 
        self.goal_dim = goal_dim
        act_space =  env.single_action_space
        obs_space = env.single_observation_space
        self.actor_critic = ActorCritic(obs_space,act_space,goal_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=self.args.policy_lr,eps=1e-5)
        self.action_space_low, self.action_space_high = torch.from_numpy(env.single_action_space.low).to(device), torch.from_numpy(env.single_action_space.high).to(device) 

        self.obs_dim = np.array(obs_space.shape).prod()
        self.act_dim = np.array(act_space.shape).prod() 
        

    def load(self,dir_path):
        ## find the model path containing actor, qf1, qf2, and load them
        for file in os.listdir(dir_path):
            if file.endswith(".actor_critic.pth"):
                actor_critic_path = os.path.join(dir_path,file)
        self.actor_critic.load_state_dict(torch.load(actor_critic_path,map_location=self.device))
        print(f"Models loaded from {dir_path}")

    def save(self):
        # Get current date and time
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")  # Folder named by the date
        timestamp_str = current_time.strftime("%H-%M-%S")  # Timestamp for the file names
        # Create a directory for the current day if it doesn't exist
        save_dir = os.path.join(self.args.save_dir,self.args.exp_name,date_str,timestamp_str)
        os.makedirs(save_dir, exist_ok=True)
        # Define file paths for actor, qf1, and qf2 within the same subfolder
        actor_critic_save_path = os.path.join(save_dir, f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}.actor_critic.pth")
        torch.save(self.actor_critic.state_dict(), actor_critic_save_path)
        # Optionally, print the paths to confirm
        print(f"Actor Critic model saved at: {actor_critic_save_path}")


    def prepare_data(self,file_dir, add_noise = False):
        self.dataset = BCDataset(file_dir,self.obs_dim,self.goal_dim,self.act_dim,add_noise)

        
    def run(self,n_iters,eval_env = None):
        """
        Interact with single environment
        """
        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        if self.args.track:
            import wandb
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )
        log_dir = os.path.join(self.args.log_dir,run_name)
        # writer = SummaryWriter(f"runs/{run_name}",log_dir = self.args.log_dir)
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
        assert hasattr(self,'dataset')
        dataloader = DataLoader(self.dataset,batch_size=self.args.batch_size,shuffle=True)

        for i in tqdm(range(n_iters), position=0, leave=True):
            info_list  = defaultdict(list)
            for obs, act ,goal in dataloader:
                self.actor_optimizer.zero_grad()
                obs = obs.to(self.device).float()
                act = act.to(self.device).float()
                goal = goal.to(self.device).float()

                _, logp, entropy = self.actor_critic.actor.get_action_dist(obs,goal,act)
                nll = -logp.mean()
                ent = entropy.mean()

                loss = nll - self.args.ent_coef * ent
                loss_info = {
                    'logp':nll.item(),
                    'entropy':ent.item()
                }
                loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step() 
                for k,v in loss_info.items():
                    info_list[k].append(v)
            
            for k,v in info_list.items():
                writer.add_scalar(f"Loss/{k}: ", np.mean(v), i)


            if (i+1) % self.args.eval_every == 0 and eval_env is not None:
                ## Eval 
                n_eval = 50 
                success_times = []
                ret_list = []
                with torch.inference_mode():
                    for _ in range(n_eval):
                        done = False 
                        obs, info = eval_env.reset()
                        ret = 0
                        while not done:
                            a = self.actor_critic.get_action(obs.to(self.device),info['goal'].to(self.device),deterministic=True)
                            obs,reward,termination,truncation,info = eval_env.step(a.cpu())
                            done = torch.logical_or(termination,truncation).to(torch.float32)
                            ret += reward

                            if torch.any(done):
                                success_times.append(info['success'][0].float())
                                ret_list.append(ret.squeeze().cpu().numpy())
                writer.add_scalar(f"Eval/Success Rate: ", np.mean(success_times),i)
                writer.add_scalar(f"Eval/Return: ", np.mean(ret_list),i)
                print(np.mean(success_times),np.mean(ret_list))
            if (i+1) % self.args.save_every_iterations == 0:
                self.save()










        
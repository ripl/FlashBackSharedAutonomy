
from .buffer import DataCollection
from ..utils.env_utils import get_frame
import random 
import  wandb
import numpy as np
import torch 
from torch.utils.tensorboard import SummaryWriter
import time 

class DataGeneration:
    """
    Env and Expert is required to be passed in order to generate data
    """
    def __init__(self, env_name, actor,args) -> None:
        """
        Args:
            env_name : str : Environment name
            expert : Actor, with agent as nn.Module 
        """
        self.env_name = env_name
        self.actor = actor
        self.args = args 

    def collect_data(self,env):
        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        if self.args.track:
            import wandb
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                dir = self.args.wandb_log_dir,
                # sync_tensorboard=False,
                # config=vars(self.args),
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )

        replay_buffer = DataCollection(
            directory=self.args.data_dir,
            chunk_size=self.args.chunk_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            exp_name=self.args.exp_name,
            goal_dim=env.goal_dim,
            non_goal_dim=env.non_goal_dim,
        )
        step = 0 
        ep = 0
        num_vis_episodes = self.args.num_vis_episodes
        scale = 0.5

        desired_num_transitions = self.args.chunk_size * self.args.desired_num_chunks
        collected_transitions = 0 
        success_ct = []
        #! DO NOT Remove this reset 
        env.reset(seed = self.args.seed)
        
        with torch.no_grad():
            while step < self.args.maximum_steps:
                if collected_transitions >= desired_num_transitions:
                    break  # Stop collecting data if we have enough transitions
                frames = []
                candidate_entries = []
                obs,info = env.reset()
                done = False
                sum_rewards = 0
                last_ep_step = 0
                sum_cost = 0.0
                # Original resolution is (400 x 600)
                if self.args.track and self.args.save_video and ep < num_vis_episodes:
                    frames.append(get_frame(env, ep, step, obs, scale=scale))
                while not done:
                    # exp_act = self.actor.act(torch.from_numpy(obs).reshape(1,-1).float().to(self.args.device),**kwargs).squeeze().cpu().numpy()
                    exp_act, _, _, _ = self.actor.step(
                        torch.from_numpy(obs).reshape(1,-1).float().to(self.args.device), 
                        deterministic=True
                        )
                    exp_act = exp_act.squeeze().cpu().numpy()
                    if random.random() < self.args.randp:
                        action = env.action_space.sample()
                        next_obs, rew,cost, done, trunctions, next_info = env.step(action)
                    else:
                        action = exp_act
                        next_obs, rew,cost, done, trunctions, next_info = env.step(action)
                        candidate_entries.append((info['non_goal_obs'], exp_act,next_info['non_goal_obs'], info['goal']))
                    

                    obs = next_obs
                    info = next_info
                    done = done or trunctions
                    step += 1
                    last_ep_step += 1
                    sum_rewards += rew
                    sum_cost += cost

                    if self.args.track and self.args.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                env,
                                ep,
                                step,
                                obs,
                                rew,
                                sum_rewards,
                                action=action,
                                scale=scale,
                            )
                        )

                if self.args.track and self.args.save_video and ep <= num_vis_episodes:
                    # Log video
                    wandb.log(
                        {   "step": ep,
                            "video": wandb.Video(
                                np.asarray(frames).transpose(0, 3, 1, 2),
                                fps=30,
                                format="mp4",
                            )
                        }
                    )

                # NOTE: only store trajectories that exceed a threshold
                # if sum_rewards >= self.args.valid_return_threshold:
                if done:
                    if sum_cost < 25.0:
                        for entry in candidate_entries:
                            obs, act, next_obs, goal = entry
                            replay_buffer.add(obs, act,next_obs, goal)
                            collected_transitions += 1
                        success_ct.append(1)
                        print(
                            f"step: {collected_transitions} / {desired_num_transitions}\tsum_rewards: {sum_rewards}\tep_len: {last_ep_step}"
                        )
                    if self.args.track:
                        wandb.log(
                            {
                                "step": step,
                                "sum_rewards": sum_rewards,
                                "sum_cost": sum_cost,   
                                "ep_len": last_ep_step,
                                "success_rate":np.mean(success_ct)
                            }
                        )

                else:
                    success_ct.append(0)
                ep += 1

        
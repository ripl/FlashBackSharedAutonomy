
from cv2 import exp
from .buffer import DataCollection
from ..utils.env_utils import get_frame

import random 
import  wandb
import numpy as np
import torch 
from torch.utils.tensorboard import SummaryWriter
import time 
from pathlib import Path
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
    
    def perturb_expert(self, env ,exp_action,method = "noisy"):
        random_action = env.action_space.sample()
        exp_action = exp_action.squeeze()
        if method == 'noisy':
            exp_action[:-1] = random_action[:-1]
            return exp_action
        elif method == 'noised':
            noise = np.random.normal(size=random_action.shape)
            noise_scale = 0.4 #! To Verify
            exp_action[:-1] = exp_action[:-1] + noise_scale * noise[:-1]
            return exp_action
        
    def collect_from_state_dict(self, env, data_dir):
        assert env.num_envs == 1, print("By default use 1 env to collect data")
        run_name = f"{self.args.env_id}__{self.args.exp_name}_d_{self.args.seed}__{int(time.time())}"
        data_directory = Path(data_dir)
        for fname in data_directory.iterdir():
            print(f"Loading Data from {fname}")
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
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            exp_name=self.args.exp_name,
            goal_dim=env.goal_dim
        )
        step = 0 
        ep = 0
        num_vis_episodes = self.args.num_vis_episodes
        scale = None

        desired_num_transitions = self.args.chunk_size * self.args.desired_num_chunks
        collected_transitions = 0 
        success_ct = []
        frames = []
        with torch.inference_mode():
            for fname in data_directory.iterdir():
                list_state_dicts = torch.load(fname)
                for state_dict in list_state_dicts:
                    candidate_entries = []
                    frames = []
                    env.reset()
                    env.set_state_dict(state_dict)
                    done = False
                    sum_rewards = 0
                    last_ep_step = 0
                    obs,info = env.get_state_info()
                    while not done:
                        if 'goal' in info:
                            kwargs = {
                            'g':info['goal'].to(self.args.device),
                            }
                        exp_act = self.actor.act(obs.to(self.args.device),**kwargs)

                        if random.random() < self.args.randp:
                            # action = env.action_space.sample()
                            action = self.perturb_expert(env,exp_act, method = 'noised')
                            next_obs, rew, done, trunctions, info = env.step(action)
                        else:
                            action = exp_act.squeeze()
                            next_obs, rew, done, trunctions, info = env.step(action)
                            candidate_entries.append((obs.cpu().numpy(), exp_act,next_obs.cpu().numpy(), info['goal']))
                        obs = next_obs
                        done = done or trunctions
                        step += 1
                        last_ep_step += 1
                        sum_rewards += rew
                        if self.args.track and self.args.save_video and ep < num_vis_episodes:
                            frames.append(
                                get_frame(
                                    env,
                                    ep,
                                    step=last_ep_step,
                                    done = done,
                                    scale=scale,
                                    info = info
                                )
                            )
                    if self.args.track and self.args.save_video and ep <= num_vis_episodes and len(frames) > 0:
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

                    if 'success' in info and info['success']:
                        # Store data only if the episode achieved sum_rewards >= threshold
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
                                    "ep_len": last_ep_step,
                                    "success_rate":np.mean(success_ct)
                                }
                            )

                    else:
                        success_ct.append(0)
                    ep += 1



    def collect_data(self,env):
        assert env.num_envs == 1, print("By default use 1 env to collect data")
        run_name = f"{self.args.env_id}__{self.args.exp_name}_d_{self.args.seed}__{int(time.time())}"
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
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            exp_name=self.args.exp_name,
            goal_dim=env.goal_dim
        )
        step = 0 
        ep = 0
        num_vis_episodes = self.args.num_vis_episodes
        scale = None

        desired_num_transitions = self.args.chunk_size * self.args.desired_num_chunks
        collected_transitions = 0 
        success_ct = []
        frames = []
        with torch.no_grad():
            while step < self.args.maximum_steps:
                if collected_transitions >= desired_num_transitions:
                    break  # Stop collecting data if we have enough transitions
                candidate_entries = []
                frames = []
                obs,info = env.reset()
                done = False
                sum_rewards = 0
                last_ep_step = 0

                if self.args.save_video and ep < num_vis_episodes:
                    frames.append(get_frame(env, ep, step=last_ep_step, scale=scale, done=done, info=info))
                while not done:
                    if 'goal' in info:
                        kwargs = {
                            'g':info['goal'].to(self.args.device),
                        }
                    exp_act = self.actor.act(obs.to(self.args.device),**kwargs)

                    if random.random() < self.args.randp:
                        # action = env.action_space.sample()
                        action = self.perturb_expert(env,exp_act, method = self.args.perturb_method)
                        next_obs, rew, done, trunctions, info = env.step(action)
                    else:
                        action = exp_act.squeeze()
                        next_obs, rew, done, trunctions, info = env.step(action)
                        candidate_entries.append((obs.cpu().numpy(), exp_act,next_obs.cpu().numpy(), info['goal']))
                    

                    obs = next_obs
                    done = done or trunctions
                    step += 1
                    last_ep_step += 1
                    sum_rewards += rew

                    if self.args.track and self.args.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                env,
                                ep,
                                step=last_ep_step,
                                scale=scale,
                                done= done,
                                info = info
                            )
                        )

                if self.args.track and self.args.save_video and ep <= num_vis_episodes and len(frames) > 0:
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
                if 'success' in info and info['success']:
                    # Store data only if the episode achieved sum_rewards >= threshold
                    for entry in candidate_entries:
                        obs, act, next_obs,goal = entry
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
                                "ep_len": last_ep_step,
                                "success_rate":np.mean(success_ct)
                            }
                        )

                else:
                    success_ct.append(0)
                ep += 1

        
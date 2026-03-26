import torch 
import numpy as np 
import gymnasium as gym 
from ..utils.env_utils import get_frame
import time
import csv
import os
import fcntl
import pandas as pd

def write_to_csv(csv_path, data_dict, cfg,method_name = None, is_ddpm = False):
    """
    Append a row to a CSV file with data from a dictionary and configuration values.
    Ensures safe access for multiple processes using file locking.
    
    Args:
        csv_path (str): Path to the CSV file.
        data_dict (dict): Dictionary with keys corresponding to CSV column names and values to write.
        cfg (object): Configuration object with attributes `expname` and `policy.noise_level`.
    """
    # Add cfg values to the dictionary
    data_dict['expname'] = method_name if method_name is not None else cfg.exp_name
    if is_ddpm:
        data_dict['fwd_diffuse_ratio'] = cfg.ddpm_policy.fwd_diffuse_ratio
    else:
        data_dict['noise_level'] = cfg.noise_level
        data_dict['n_steps'] = cfg.n_steps
        data_dict['noise_ratio'] = cfg.noise_level / cfg.n_steps
    data_dict['actor_type'] = cfg.actor_type
    data_dict['seed'] = cfg.seed
    for key, value in cfg.actor_kwargs.items():
        data_dict[f'actor_{key}'] = value

    # Define headers from the data_dict
    # headers = data_dict.keys()
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as csvfile:
            pass  # Simply create the file

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='a+', newline='') as csvfile:
        fcntl.flock(csvfile, fcntl.LOCK_EX)
        try:
            csvfile.seek(0)
            try:
                df_old = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                df_old = pd.DataFrame()

            df_new = pd.DataFrame([data_dict])
            df_combined = pd.concat([df_old, df_new], ignore_index=True, sort=False)
            df_combined.to_csv(csv_path, index=False)
        finally:
            fcntl.flock(csvfile, fcntl.LOCK_UN)



class Evaluator:
    def __init__(self,
                 cfg,
                 env,
                 raw_actor,
                 assistive_actor,
                 num_episode,
                 max_episode_step,
                 device) -> None:
        self.cfg = cfg
        self.env = env ## this env is dedicated for evaluation
        self.raw_actor = raw_actor 
        self.assitive_actor = assistive_actor
        self.num_episode = num_episode
        self.max_episode_step = max_episode_step
        self.device = device 


        self.raw_actor_performance = 0.0
        self.assitive_actor_performance = 0.0
    
    def evaluate_raw_actor(self):
        act_fn = lambda obs, kwargs: self.raw_actor.act(obs,**kwargs)
        res = self.rollout(act_fn,rollout_name="Raw")
        return res

    def evalute_assitive_actor(self):
        act_fn = lambda obs,kwargs: self.assitive_actor.act(obs,**kwargs)

        res = self.rollout(act_fn,rollout_name="Assitive")
        return res 

    def evaluate_blend_actor(self):
        act_fn = lambda obs,kwargs: self.assitive_actor.blend_act(obs, beta=self.cfg.beta, **kwargs)
        res = self.rollout(act_fn,rollout_name="Blend")
        return res 


    def rollout(self, actor_fn, rollout_name =None):
        """
        actor_fn: 
            input: observation 
            output: action
        """
        run_name = f"{self.cfg.env_id}_{self.cfg.actor_type}_{self.cfg.exp_name}_{self.cfg.seed}_{int(time.time())}"
        if self.cfg.track:
            import wandb
            from omegaconf import OmegaConf
            wandb.init(
                project=self.cfg.wandb_project_name,
                entity=self.cfg.wandb_entity,
                dir = self.cfg.wandb_log_dir,
                # sync_tensorboard=False,
                # config=self.cfg,
                name=run_name,
                monitor_gym=False,
                save_code=True,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
        num_vis_episodes = 10
        scale = 0.5

        list_reward = []
        list_episode_length = []
        num_crash = 0
        num_success = 0
        num_time_out = 0
        num_out_of_bound = 0

        ## Important to reproduce the result
        self.env.reset(seed = self.cfg.seed)

        # generate a fixed seed list
        rng = np.random.default_rng(self.cfg.seed)
        seed_list = rng.integers(low=0, high=2**32 - 1, size=self.num_episode).tolist()

        with torch.no_grad():
            for ep in range(self.num_episode):
                frames = []
                obs,info = self.env.reset(seed = seed_list[ep])
                rewards,lengths = 0,0

                if self.cfg.track and self.cfg.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                self.env,
                                ep,
                                0,
                                obs,
                                scale=scale,
                                exp_name=rollout_name
                            )
                        )
                for step in range(self.max_episode_step):
                    if 'goal' in info:
                        kwargs = {'g':info['goal']}
                    action = actor_fn(obs,kwargs)
                    obs, rew, done, trunctions, info = self.env.step(action)
                    rewards += rew
                    if self.cfg.track and self.cfg.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                self.env,
                                ep,
                                step,
                                obs,
                                rew,
                                rewards,
                                action=action,
                                scale=scale,
                                exp_name=rollout_name
                            )
                        )

                    lengths += 1
                    if done:
                        list_reward.append(rewards)
                        list_episode_length.append(lengths)
                        termination_info = info['termination_info']
                        if termination_info == 'target_reached' or termination_info == 'landed_at_site':
                            num_success += 1
                        elif termination_info == 'crashed':
                            num_crash += 1
                        elif termination_info == 'time_out':
                            num_time_out += 1
                        elif termination_info == 'out_of_bound':
                            num_out_of_bound += 1
                        break 



                # if self.cfg.save_video and ep == min(num_vis_episodes - 1,self.num_episode-1):
                if self.cfg.track and self.cfg.save_video and ep <= min(num_vis_episodes - 1,self.num_episode-1):
                    # Log video
                    wandb.log(
                        {
                            "video": wandb.Video(
                                np.asarray(frames).transpose(0, 3, 1, 2),
                                fps=30,
                                format="mp4",
                            )

                        }
                    )
                
                # if self.cfg.save_video and (ep < self.num_episode):
                if self.cfg.track:
                    wandb.log(
                        {
                            "ep": ep,
                            "sum_rewards": rewards,
                            "ep_len": lengths,
                            "success_rate": num_success / (ep+1),
                            "crash_rate": num_crash / (ep+1),
                            "time_out_rate": num_time_out / (ep+1),
                            "out_of_bound_rate": num_out_of_bound / (ep+1),
                            }
                        )
        
                
        success_rate = num_success / self.num_episode
        crash_rate = num_crash / self.num_episode
        time_out_rate = num_time_out / self.num_episode
        out_of_bound_rate = num_out_of_bound / self.num_episode
        res = {
            "success_rate":success_rate,
            "crash_rate":crash_rate,
            "time_out_rate":time_out_rate,
            "out_of_bound_rate":out_of_bound_rate,
            "Return_mean":np.mean(list_reward),
            "Return_std":np.std(list_reward),
            "Episode_length_mean":np.mean(list_episode_length),
            "Episode_length_std":np.std(list_episode_length)
        }

        return res


class Maniskill_Evaluator:
    def __init__(self,
                 cfg,
                 env,
                 raw_actor,
                 assistive_actor,
                 num_episode,
                 max_episode_step,
                 device) -> None:
        self.cfg = cfg
        self.env = env ## this env is dedicated for evaluation
        self.raw_actor = raw_actor 
        self.assitive_actor = assistive_actor
        self.num_episode = num_episode
        self.max_episode_step = max_episode_step
        self.device = device 


        self.raw_actor_performance = 0.0
        self.assitive_actor_performance = 0.0
    
    def evaluate_raw_actor(self):
        act_fn = lambda obs, kwargs: self.raw_actor.act(obs,**kwargs)
        res = self.rollout(act_fn,rollout_name="Raw")
        return res

    def evalute_assitive_actor(self):
        act_fn = lambda obs,kwargs: self.assitive_actor.act(obs,**kwargs)

        res = self.rollout(act_fn,rollout_name="Assitive")
        return res 

    def evaluate_blend_actor(self):
        act_fn = lambda obs,kwargs: self.assitive_actor.blend_act(obs, beta=self.cfg.beta, **kwargs)
        res = self.rollout(act_fn,rollout_name="Blend")
        return res 


    def rollout(self, actor_fn,rollout_name =None):
        """
        actor_fn: 
            input: observation 
            output: action
        """
        run_name = f"{self.cfg.env_id}_{self.cfg.actor_type}_{self.cfg.exp_name}_{self.cfg.seed}_{int(time.time())}"
        if self.cfg.track:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project_name,
                entity=self.cfg.wandb_entity,
                dir = self.cfg.wandb_log_dir,
                # sync_tensorboard=False,
                # config=vars(self.args),
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )
        num_vis_episodes = 10
        scale = 1.0

        list_reward = []
        list_episode_length = []
        num_success = 0
        num_time_out = 0

        ## Important to reproduce the result
        self.env.reset(seed = self.cfg.seed)
        # generate a fixed seed list
        rng = np.random.default_rng(self.cfg.seed)
        seed_list = rng.integers(low=0, high=2**32 - 1, size=self.num_episode).tolist()
        frames = []
        with torch.no_grad():
            for ep in range(self.num_episode):
                
                # obs,info = self.env.reset(seed = self.cfg.seed)
                obs,info = self.env.reset(seed = seed_list[ep])
                rewards,lengths = 0,0

                if self.cfg.track and self.cfg.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                self.env,
                                ep,
                                0,
                                scale=scale,
                            )
                        )
                for step in range(self.max_episode_step):
                    if 'goal' in info:
                        kwargs = {'g':info['goal']}
                    action = actor_fn(obs,kwargs)
                    obs, rew, done, trunctions, info = self.env.step(action)
                    rewards += rew
                    done = torch.logical_or(done, trunctions)
                    if self.cfg.track and self.cfg.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                self.env,
                                ep,
                                step,
                                scale=scale,
                                exp_name=rollout_name
                            )
                        )

                    lengths += 1

                    if done:
                        list_reward.append(rewards)
                        list_episode_length.append(lengths)
                        if 'success' in info.keys() and info['success']:
                            num_success += 1
                        else:
                            num_time_out += 1
                        break 
                if self.cfg.track and self.cfg.save_video and ep <= min(num_vis_episodes - 1,self.num_episode-1):
                    # Log video
                    wandb.log(
                        {
                            "video": wandb.Video(
                                np.asarray(frames).transpose(0, 3, 1, 2),
                                fps=30,
                                format="mp4",
                            )

                        }
                    )
                # if self.cfg.save_video and (ep < self.num_episode):
                if self.cfg.track:
                    wandb.log(
                        {
                            "ep": ep,
                            "sum_rewards": rewards,
                            "ep_len": lengths,
                            "success_rate": num_success / (ep+1),
                            "time_out_rate": num_time_out / (ep+1),
                            }
                        )
    
                
        success_rate = num_success / self.num_episode
        time_out_rate = num_time_out / self.num_episode

        # breakpoint()

        # On elm, GPU
        # list_reward = torch.stack(list_reward)
        res = {
            "success_rate":success_rate,
            "time_out_rate":1-success_rate,
            # "Return_mean":list_reward.mean().item(),
            # "Return_std":list_reward.std().item(),
            "Episode_length_mean":np.mean(list_episode_length),
            "Episode_length_std":np.std(list_episode_length)
        }

        return res


class SafetyGym_Evaluator:
    def __init__(self,
                 cfg,
                 env,
                 raw_actor,
                 assistive_actor,
                 num_episode,
                 max_episode_step,
                 device) -> None:
        self.cfg = cfg
        self.env = env ## this env is dedicated for evaluation
        self.raw_actor = raw_actor 
        self.assitive_actor = assistive_actor
        self.num_episode = num_episode
        self.max_episode_step = max_episode_step
        self.device = device 


        self.raw_actor_performance = 0.0
        self.assitive_actor_performance = 0.0
    
    def evaluate_raw_actor(self):
        act_fn = lambda obs, kwargs: self.raw_actor.act(obs)
        res = self.rollout(act_fn,rollout_name="Raw")
        return res

    def evalute_assitive_actor(self):
        def act_fn(obs,info): 
            user_action = self.assitive_actor.get_user_action(obs)
            action = self.assitive_actor.get_diffusion_action(info['non_goal_obs'],user_action)
            return action
        res = self.rollout(act_fn,rollout_name="Assitive")
        return res 


    def rollout(self, actor_fn,rollout_name =None):
        """
        actor_fn: 
            input: observation 
            output: action
        """
        run_name = f"{self.cfg.env_id}_{self.cfg.actor_type}_{self.cfg.exp_name}_{self.cfg.seed}_{int(time.time())}"
        if self.cfg.track:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project_name,
                entity=self.cfg.wandb_entity,
                dir = self.cfg.wandb_log_dir,
                # sync_tensorboard=False,
                # config=vars(self.args),
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )
        num_vis_episodes = 10
        scale = 1.0

        list_reward = []
        list_episode_length = []
        list_cost = []
        list_success = []
        

        ## Important to reproduce the result
        self.env.reset(seed = self.cfg.seed)
        frames = []
        with torch.no_grad():
            for ep in range(self.num_episode):
                
                obs,info = self.env.reset(seed = self.cfg.seed)
                rewards,costs,lengths = 0,0,0

                if self.cfg.track and self.cfg.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                self.env,
                                ep,
                                0,
                                scale=scale,
                            )
                        )
                for step in range(self.max_episode_step):
                    action = actor_fn(obs,info)
                    obs, rew, cost, done, trunctions, info = self.env.step(action)
                    rewards += rew
                    costs += cost
                    done = done | trunctions
                    if self.cfg.track and self.cfg.save_video and ep < num_vis_episodes:
                        frames.append(
                            get_frame(
                                self.env,
                                ep,
                                step,
                                scale=scale,
                                exp_name=rollout_name
                            )
                        )

                    lengths += 1
                    if done:
                        list_reward.append(rewards)
                        list_episode_length.append(lengths)
                        list_cost.append(costs)
                        if costs <= 25.0:
                            list_success.append(1)
                        else:
                            list_success.append(0)
        
                if self.cfg.track and self.cfg.save_video and ep <= min(num_vis_episodes - 1,self.num_episode-1):
                    # Log video
                    wandb.log(
                        {
                            "video": wandb.Video(
                                np.asarray(frames).transpose(0, 3, 1, 2),
                                fps=30,
                                format="mp4",
                            )

                        }
                    )
                # if self.cfg.save_video and (ep < self.num_episode):
                if self.cfg.track:
                    wandb.log(
                        {
                            "ep": ep,
                            "ep_len": lengths,
                            "Ep_Reward": np.mean(list_reward),
                            "Ep_Cost": np.mean(list_cost),
                            "Success Rate": np.mean(list_success)
                            }
                        )
        
        res = {
            "Ep_Reward": np.mean(list_reward),
            "Ep_Reward_std": np.std(list_reward),
            "Ep_Cost": np.mean(list_cost),
            "Ep_Cost_std": np.std(list_cost),
            "Success Rate": np.mean(list_success),
            "Episode_length_mean":np.mean(list_episode_length),
            "Episode_length_std":np.std(list_episode_length)
        }

        return res
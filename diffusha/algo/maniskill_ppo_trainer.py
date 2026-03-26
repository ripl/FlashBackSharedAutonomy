from pyparsing import makeXMLTags
import torch 
from ..model.ppo import ActorCritic_v2,ActorCritic_v3
from ..model.diffusion_model import ForwardModel
from .algo_utils import decompose_state_dicts
import time 
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from diffusha.data_collection.buffer import RolloutBuffer
from torch.utils.data import DataLoader
import os
from datetime import datetime
from collections import deque,defaultdict
import torch.nn as nn 
from diffusha.algo.algo_utils import combined_cosine_polynomial_decay
    
class ExpertTrainer:
    def __init__(self, env,args):
        self.args = args
      
        device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.device = device
        goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
        self.num_envs = env.unwrapped.num_envs 
        self.goal_dim = goal_dim
        act_space =  env.single_action_space
        obs_space = env.single_observation_space
        self.actor_critic = ActorCritic_v3(obs_space,act_space,goal_dim,self.args.action_rescale).to(device)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=self.args.policy_lr,eps=1e-5)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=self.args.critic_lr,eps=1e-5)
        self.action_space_low, self.action_space_high = torch.from_numpy(env.single_action_space.low).to(device), torch.from_numpy(env.single_action_space.high).to(device)
        
        self.rb = RolloutBuffer(
            num_envs=self.num_envs,
            num_transitions_per_env=self.args.num_steps_per_env,
            obs_shape=obs_space.shape,
            act_shape=act_space.shape,
            goal_dim=goal_dim,
            device=self.device
        )

        self.return_queue = deque(maxlen=50)
        self.ep_len_queue = deque(maxlen=50)
        self.success_queue = deque(maxlen=50)
        self.cum_rewards = torch.zeros((self.num_envs)).to(device)
        self.ep_lens = torch.zeros((self.num_envs)).to(device)
        
        ## Training Trick
        self.ent_coef = self.args.ent_coef
        self.ent_coef_decay = self.args.ent_coef_decay
        if self.ent_coef_decay:
            self.max_ent_coef = self.ent_coef
            self.min_ent_coef = 0.0 
        else:
            self.max_ent_coef, self.min_ent_coef = self.ent_coef,self.ent_coef

        self.num_save_data = args.num_save_data
        self.num_save_data_every = args.num_save_data_every
        if self.num_save_data > 0:
            self.tmp_save_data_list = deque(maxlen=self.num_save_data_every)
        
        self.best_success = 0.0
    
    def save_state_dict_data(self,env):
        if self.num_save_data > 0:
            is_grasp_env_idx = env.is_grasp_peg

            tmp_state_list = decompose_state_dicts(env.get_state_dict(),is_grasp_env_idx)
            self.tmp_save_data_list.extend(tmp_state_list)
            if len(self.tmp_save_data_list) == self.num_save_data_every:
                save_dir = os.path.join(self.args.save_dir,self.args.exp_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"data_{self.num_save_data}.pt")
                torch.save(list(self.tmp_save_data_list),save_path )
                self.num_save_data -= self.num_save_data_every
                self.tmp_save_data_list.clear()
                print("Save Data For Replay")
        else:
            pass 


    def compute_ppo_loss(self,obs,act,goal,logprobs,advantages,returns,values):
        _,newlogprob,entropy,newvalue = self.actor_critic.get_action_and_value(obs,goal,act)
        logratio = newlogprob - logprobs
        ratio = logratio.exp() 
        with torch.no_grad():
            old_approx_kl = (-logratio).mean() 
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
        pg_loss1 = -advantages * ratio 
        pg_loss2 = -advantages * torch.clamp(ratio, 1-self.args.clip_coef, 1 + self.args.clip_coef)
        pg_loss = torch.max(pg_loss1,pg_loss2).mean() 

        ## Value loss 
        if self.args.clip_vloss:
            v_loss_unclipped = (newvalue - returns)**2 
            v_clipped = values + torch.clamp(newvalue - values, -self.args.clip_coef,self.args.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2 
            v_loss_max = torch.max(v_loss_clipped,v_loss_unclipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
        # import pdb; pdb.set_trace()
        ## Entropy 
        entropy_loss = entropy.mean() 
        actor_loss = pg_loss - self.ent_coef * entropy_loss 
        critic_loss = v_loss * self.args.vf_coef 
        loss_info = {
            'pg_loss':pg_loss.item(),
            'v_loss':v_loss.item(),
            'ent':entropy.mean().item(),
            'approx_kl':approx_kl.item(),
            "clipfracs":clipfracs,
            'average_return':returns.mean().item(),
            'average_values':values.mean().item()

        }
        return actor_loss,critic_loss, loss_info
    
    def update_penalty_clip_ratio(self,success_rate, cur_iter, max_iteration, min_ratio, max_ratio ):
        if success_rate < 0.8:
            ratio = min_ratio + (max_ratio - min_ratio) * np.clip((cur_iter / max_iteration / 2),0,1)
        else:
            ratio = max_ratio
        return ratio


    def run(self, env, forward_trainer = None):
        """
        Interact with single environment
        """
        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}"
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

        def clip_action(action: torch.Tensor):
            return torch.clamp(action.detach(), self.action_space_low, self.action_space_high)
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        obs, info = env.reset(seed = self.args.seed)
        dones = torch.zeros(self.num_envs, device = self.device)
        goals = info['goal']
        for iteration in range(1, self.args.total_iterations):
            self.single_step_rewards = torch.zeros((self.num_envs)).to(self.device)
            self.single_step_penaltys = torch.zeros((self.num_envs)).to(self.device)
            
            with torch.inference_mode():
                for step in range(0, self.args.num_steps_per_env):
                    global_step += self.num_envs

                    actions, logprob, _, value = self.actor_critic.get_action_and_value(obs, goals)
                

                    next_obs, reward, termination, truncation, info = env.step(clip_action(actions))
                    ## beta version 
                    self.single_step_rewards += info['raw_reward']
                    self.single_step_penaltys += info['penalty']

                    next_dones,next_goals = torch.logical_or(termination ,truncation).to(torch.float32), info['goal']  ### any dones's agent's next obs is wrong ? No, not for truncation
                    
                    self.rb.add_transitions(obs,actions,info['goal'],reward,dones,logprob,value)
                    obs,goals = next_obs,next_goals
                    dones = next_dones
                    self.cum_rewards += reward
                    self.ep_lens += 1 

                    if torch.any(dones):
                        reset_index = torch.where(dones)[0]
                        success = info['success'][reset_index].float().tolist()
                        final_values = self.actor_critic.get_value(next_obs,info['goal'])
                        self.rb.add_final_values(step, reset_index, final_values)

                        obs, info = env.reset(options = {
                            'env_idx':reset_index
                        })
                        goals = info['goal']
                        ret = self.cum_rewards[reset_index].tolist()
                        ep_len = self.ep_lens[reset_index].tolist()
                        self.success_queue.extend(success)
                        self.return_queue.extend(ret)
                        self.ep_len_queue.extend(ep_len)
                        self.cum_rewards[reset_index] = 0 
                        self.ep_lens[reset_index] = 0 

            ### Logging
            writer.add_scalar("evaluation/return_mean", np.mean(self.return_queue), global_step)
            writer.add_scalar("evaluation/return_std", np.std(self.return_queue), global_step)
            writer.add_scalar("evaluation/episode_length", np.mean(self.ep_len_queue), global_step)
            writer.add_scalar("evaluation/success_rate", np.mean(self.success_queue), global_step)

            writer.add_scalar("evaluation/reward_per_iter", torch.mean(self.single_step_rewards).detach().cpu().numpy(), global_step)
            writer.add_scalar("evaluation/penalty_per_iter", torch.mean(self.single_step_penaltys).detach().cpu().numpy(), global_step)
            writer.add_scalar("evaluation/ratio_penalty", torch.mean(self.single_step_penaltys/self.single_step_rewards).detach().cpu().numpy(), global_step)

            success_rate= np.mean(self.success_queue)
            
            if success_rate > self.best_success:
                self.best_success = success_rate
                self.save(suffix='best')
                print("Current best model saved, with success rate: ", success_rate)
            
            ### Save data for 
            if success_rate > 0.2:
                self.save_state_dict_data(env)
            
            ### PPO algorithm
            with torch.no_grad():
                last_value = self.actor_critic.get_value(next_obs,next_goals)
                self.rb.compute_returns(last_value,next_dones,self.args.gamma,self.args.lam) 

            generator = self.rb.mini_batch_genetor(num_mini_batch=self.args.num_mini_batch,
                                                   num_epochs=self.args.num_training_epochs)
            
            info_list  = defaultdict(list)
            training_ct=0
            skip_actor_loss = False
            for samples in generator:
                obs_batch,act_batch,goal_batch,val_batch,ret_batch,logp_batch,logp_batch,adv_batch = samples 
                actor_loss,critic_loss,loss_info = self.compute_ppo_loss(
                    obs = obs_batch,
                    act = act_batch,
                    goal= goal_batch,
                    logprobs= logp_batch,
                    advantages=adv_batch,
                    returns=ret_batch,
                    values=val_batch
                )
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                if skip_actor_loss:
                    actor_loss.detach()
                else:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.args.max_grad_norm)
                    self.actor_optimizer.step()
                for k,v in loss_info.items():
                    info_list[k].append(v)
                training_ct += 1 
                if training_ct % self.args.num_mini_batch == 0:
                    ## Full training 
                    if self.args.target_kl is not None and  loss_info['approx_kl'] > self.args.target_kl:
                        skip_actor_loss = True
                    
            for k,v in info_list.items():
                writer.add_scalar(f"losses/{k}: ", np.mean(v), global_step)
            
            ### Save Model 
            if iteration % self.args.save_every_iterations == 0:
                self.save()
            
            if self.ent_coef_decay:
                self.ent_coef = combined_cosine_polynomial_decay(self.max_ent_coef,self.min_ent_coef,iteration,self.args.total_iterations)

            ### change parameter
            # env.set_penalty_clip(ratio = self.update_penalty_clip_ratio(success_rate, iteration,max_iteration=self.args.total_iterations,
            #                                                             min_ratio=0.1, max_ratio=1.0))
    

    def CL_run(self, env, forward_trainer = None):
        """
        Interact with single environment
        """
        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}"
        env.set_initial_eval_threshold(self.args.initial_eval_threshold)
        env.unwrapped.update_curriculum(0)
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
            config_dict = {}
            for k, v in vars(self.args).items():
                try:
                    import json
                    json.dumps(v)
                    config_dict[k] = v
                except (TypeError, OverflowError):
                    config_dict[k] = str(v)
            wandb.config.update(config_dict)
        log_dir = os.path.join(self.args.log_dir,run_name)
        # writer = SummaryWriter(f"runs/{run_name}",log_dir = self.args.log_dir)
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        def clip_action(action: torch.Tensor):
            return torch.clamp(action.detach(), self.action_space_low, self.action_space_high)
        # TRY NOT TO MODIFY: start the game
        
        global_step = 0
        obs, info = env.reset(seed = self.args.seed)
        dones = torch.zeros(self.num_envs, device = self.device)
        goals = info['goal']
        curriculum_start = self.args.curriculum_start * self.args.total_iterations
        curriculum_end = self.args.curriculum_end * self.args.total_iterations
        for iteration in range(1, self.args.total_iterations):
            
            # record current curriculum progress
            if iteration > curriculum_start and iteration <= curriculum_end:
                progress = min((iteration - curriculum_start) / (curriculum_end - curriculum_start),1.0)
                # update the curriculum in the environment
                env.unwrapped.update_curriculum(progress)

            self.single_step_rewards = torch.zeros((self.num_envs)).to(self.device)
            self.single_step_penaltys = torch.zeros((self.num_envs)).to(self.device)
            
            with torch.inference_mode():
                for step in range(0, self.args.num_steps_per_env):
                    global_step += self.num_envs

                    actions, logprob, _, value = self.actor_critic.get_action_and_value(obs, goals)
                

                    next_obs, reward, termination, truncation, info = env.step(clip_action(actions))
                    ## beta version 
                    self.single_step_rewards += info['raw_reward']
                    self.single_step_penaltys += info['penalty']

                    next_dones,next_goals = torch.logical_or(termination ,truncation).to(torch.float32), info['goal']  ### any dones's agent's next obs is wrong ? No, not for truncation
                    
                    self.rb.add_transitions(obs,actions,info['goal'],reward,dones,logprob,value)
                    obs,goals = next_obs,next_goals
                    dones = next_dones
                    self.cum_rewards += reward
                    self.ep_lens += 1 

                    if torch.any(dones):
                        reset_index = torch.where(dones)[0]
                        success = info['success'][reset_index].float().tolist()
                        final_values = self.actor_critic.get_value(next_obs,info['goal'])
                        self.rb.add_final_values(step, reset_index, final_values)

                        obs, info = env.reset(options = {
                            'env_idx':reset_index
                        })
                        goals = info['goal']
                        ret = self.cum_rewards[reset_index].tolist()
                        ep_len = self.ep_lens[reset_index].tolist()
                        self.success_queue.extend(success)
                        self.return_queue.extend(ret)
                        self.ep_len_queue.extend(ep_len)
                        self.cum_rewards[reset_index] = 0 
                        self.ep_lens[reset_index] = 0 

            ### Logging
            writer.add_scalar("evaluation/return_mean", np.mean(self.return_queue), global_step)
            writer.add_scalar("evaluation/return_std", np.std(self.return_queue), global_step)
            writer.add_scalar("evaluation/episode_length", np.mean(self.ep_len_queue), global_step)
            writer.add_scalar("evaluation/success_rate", np.mean(self.success_queue), global_step)

            writer.add_scalar("evaluation/reward_per_iter", torch.mean(self.single_step_rewards).detach().cpu().numpy(), global_step)
            writer.add_scalar("evaluation/penalty_per_iter", torch.mean(self.single_step_penaltys).detach().cpu().numpy(), global_step)
            writer.add_scalar("evaluation/ratio_penalty", torch.mean(self.single_step_penaltys/self.single_step_rewards).detach().cpu().numpy(), global_step)

            success_rate= np.mean(self.success_queue)

            if self.args.track:
                # assume all env have same clearance
                current_clearance = env.unwrapped.curriculum_eval_threshold
                wandb.log({
                    "evaluation/curriculum_threshold": current_clearance,
                    "evaluation/success_rate": np.mean(self.success_queue),
                    "global_step": global_step
                })
            
            if success_rate > self.best_success:
                self.best_success = success_rate
                self.save(suffix='best')
                print("Current best model saved, with success rate: ", success_rate)
            
            ### Save data for 
            if success_rate > 0.2:
                self.save_state_dict_data(env)
            
            ### PPO algorithm
            with torch.no_grad():
                last_value = self.actor_critic.get_value(next_obs,next_goals)
                self.rb.compute_returns(last_value,next_dones,self.args.gamma,self.args.lam) 

            generator = self.rb.mini_batch_genetor(num_mini_batch=self.args.num_mini_batch,
                                                   num_epochs=self.args.num_training_epochs)
            
            info_list  = defaultdict(list)
            training_ct=0
            skip_actor_loss = False
            for samples in generator:
                obs_batch,act_batch,goal_batch,val_batch,ret_batch,logp_batch,logp_batch,adv_batch = samples 
                actor_loss,critic_loss,loss_info = self.compute_ppo_loss(
                    obs = obs_batch,
                    act = act_batch,
                    goal= goal_batch,
                    logprobs= logp_batch,
                    advantages=adv_batch,
                    returns=ret_batch,
                    values=val_batch
                )
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                if skip_actor_loss:
                    actor_loss.detach()
                else:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.args.max_grad_norm)
                    self.actor_optimizer.step()
                for k,v in loss_info.items():
                    info_list[k].append(v)
                training_ct += 1 
                if training_ct % self.args.num_mini_batch == 0:
                    ## Full training 
                    if self.args.target_kl is not None and  loss_info['approx_kl'] > self.args.target_kl:
                        skip_actor_loss = True
                    
            for k,v in info_list.items():
                writer.add_scalar(f"losses/{k}: ", np.mean(v), global_step)
            
            ### Save Model 
            if iteration % self.args.save_every_iterations == 0:
                self.save()
            
            if self.ent_coef_decay:
                self.ent_coef = combined_cosine_polynomial_decay(self.max_ent_coef,self.min_ent_coef,iteration,self.args.total_iterations)

            ### change parameter
            # env.set_penalty_clip(ratio = self.update_penalty_clip_ratio(success_rate, iteration,max_iteration=self.args.total_iterations,
            #                                                             min_rati
            
    def load(self,dir_path):
        ## find the model path containing actor, qf1, qf2, and load them
        for file in os.listdir(dir_path):
            if file.endswith(".actor_critic.pth"):
                actor_critic_path = os.path.join(dir_path,file)
        self.actor_critic.load_state_dict(torch.load(actor_critic_path,map_location=self.device))
        print(f"Models loaded from {dir_path}")

    def save(self, suffix=None):
        # Get current date and time
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")  # Folder named by the date
        timestamp_str = current_time.strftime("%H-%M-%S")  # Timestamp for the file names
        # Create a directory for the current day if it doesn't exist
        save_dir = os.path.join(self.args.save_dir, self.args.exp_name, date_str, timestamp_str)
        if suffix is not None:
            save_dir = os.path.join(self.args.save_dir, self.args.exp_name, suffix)
        os.makedirs(save_dir, exist_ok=True)

        # Define file paths for actor, qf1, and qf2 within the same subfolder
        actor_critic_save_path = os.path.join(save_dir, f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}.actor_critic.pth")
        torch.save(self.actor_critic.state_dict(), actor_critic_save_path)
        # Optionally, print the paths to confirm
        print(f"Actor Critic model saved at: {actor_critic_save_path}")
      

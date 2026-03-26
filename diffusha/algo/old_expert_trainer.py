import torch 
from ..model.sac import Actor,SoftQNetwork
from diffusha.actor.base import ExpertActor
import time 
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ..data_collection.buffer import ReplayBuffer
from ..utils.eval_utils import Evaluator
import os
from datetime import datetime
from argparse import Namespace

class ExpertTrainer:
    def __init__(self, env, args):
        self.args = args
      
        device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.device = device
        goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
        self.goal_dim = goal_dim
        act_space, obs_space = env.action_space, env.observation_space
        self.actor = Actor(obs_space,act_space,goal_dim).to(device)
        self.qf1 = SoftQNetwork(obs_space,act_space,goal_dim).to(device)
        self.qf2 = SoftQNetwork(obs_space,act_space,goal_dim).to(device)
        self.qf1_target = SoftQNetwork(obs_space,act_space,goal_dim).to(device)
        self.qf2_target = SoftQNetwork(obs_space,act_space,goal_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr)

        self.best_success = 0.0

        # Automatic entropy tuning
        if self.args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha
        
        self.rb = ReplayBuffer(
            self.args.buffer_size,
            env.observation_space,
            env.action_space,
            device,
            handle_timeout_termination=False,
            goal_dim=goal_dim,
        )

        self.fake_eval_cfg = Namespace(
            env_id=self.args.env_id,
            actor_type="training expert",
            exp_name="test_experiment",
            seed=42,
            wandb_project_name="test_project",
            wandb_entity="test_entity",
            wandb_log_dir="./wandb_logs",
            track=False,
            save_video=False,
            max_episode_steps=1000,
            num_episode=30,
        )

    def save(self, suffix = None):
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
        actor_save_path = os.path.join(save_dir, f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}.actor.pth")
        qf1_save_path = os.path.join(save_dir, f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}.qf1.pth")
        qf2_save_path = os.path.join(save_dir, f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}.qf2.pth")

        torch.save(self.actor.state_dict(), actor_save_path)
        torch.save(self.qf1.state_dict(), qf1_save_path)
        torch.save(self.qf2.state_dict(), qf2_save_path)

        # Optionally, print the paths to confirm
        print(f"Actor model saved at: {actor_save_path}")
        print(f"QF1 model saved at: {qf1_save_path}")
        print(f"QF2 model saved at: {qf2_save_path}")

        
    def run(self, env):
        """
        Interact with single environment
        """
        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.task}__{self.args.seed}__{int(time.time())}"
        log_dir = os.path.join(self.args.log_dir,run_name)
        # writer = SummaryWriter(f"runs/{run_name}",log_dir = self.args.log_dir)
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        start_time = time.time()
        # TRY NOT TO MODIFY: start the game
        obs,info = env.reset(seed=self.args.seed)
        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.args.learning_starts:
                actions = np.array(env.action_space.sample())
            else:
                actions, _, _ = self.actor.get_action(x = torch.Tensor(obs).to(self.device).reshape(1, -1),
                                                       g = torch.Tensor(info['goal']).to(self.device).reshape(1, -1),
                                                       deterministic=False)
                actions = actions.squeeze().detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncation, info = env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "episode" in info:
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            # for idx, trunc in enumerate(truncations):
            #     if trunc:
            #         print(infos.keys())
            #         real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, reward, termination, info)
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if termination or truncation:
                obs, _ = env.reset()

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                data = self.rb.sample(self.args.batch_size)
                if self.goal_dim > 0:
                    goal = data.goals
                else:
                    goal = None
                    
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations, goal)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions, goal)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions, goal)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions,goal).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions,goal).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations, goal)
                        qf1_pi = self.qf1(data.observations, pi, goal)
                        qf2_pi = self.qf2(data.observations, pi, goal)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations, goal)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", self.alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if self.args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)


            # evaluate the policy during training
            if (global_step+1)%10000 ==0:
                eval_actor = ExpertActor(
                    obs_space=env.observation_space,
                    act_space=env.action_space,
                    expert_agent=self.actor,
                    goal_dim=self.goal_dim,
                    device=self.device
                )
                # Define evaluator
                evaluator = Evaluator(
                    cfg=self.fake_eval_cfg,
                    env=env,
                    raw_actor=eval_actor,
                    assistive_actor=None,
                    num_episode=self.fake_eval_cfg.num_episode,
                    max_episode_step=self.fake_eval_cfg.max_episode_steps,
                    device=self.device,
                )
                # Run evaluation
                eval_results = evaluator.evaluate_raw_actor()
                
                # Log to wandb / tensorboard
                success_rate = eval_results["success_rate"]
                crash_rate = eval_results["crash_rate"]
                time_out_rate = eval_results["time_out_rate"]
                out_of_bound_rate = eval_results["out_of_bound_rate"]
                return_mean = eval_results["Return_mean"]
                return_std = eval_results["Return_std"]

                writer.add_scalar("evaluation/success_rate", success_rate, global_step)
                writer.add_scalar("evaluation/crash_rate", crash_rate, global_step)
                writer.add_scalar("evaluation/time_out_rate", time_out_rate, global_step)
                writer.add_scalar("evaluation/out_of_bound_rate", out_of_bound_rate, global_step)
                writer.add_scalar("evaluation/return_mean", return_mean, global_step)
                writer.add_scalar("evaluation/return_std", return_std, global_step)

                if success_rate > self.best_success:
                    self.save(suffix="best")
                    self.best_success = success_rate
                    print("Current best model saved, with success rate: ", success_rate)
            
            if (global_step+1)%100000 ==0:
                self.save()
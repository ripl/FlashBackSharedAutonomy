import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import multiprocessing
from itertools import product
import pandas as pd
from diffusha.model.ppo import ActorCritic_v3
from omegaconf import DictConfig, OmegaConf
from diffusha.actor.base import NoisyActor,LaggyActor,RandomActor,ExpertActor,NoisedActor,SlowActor
from diffusha.utils.env_utils import set_random_seed
from diffusha.utils.env_utils import make_env

from diffusha.utils.eval_utils import Evaluator
from diffusha.model.sac import Actor
from diffusha.model.model_utils import load_model
import gymnasium as gym 

def build_actor(actor_type, env, expert_actor, goal_dim, **kwargs):
    assert actor_type in ['laggy', 'noisy', 'random', 'noised', 'slow']
    if actor_type == 'laggy':
        actor = LaggyActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **kwargs)
    elif actor_type == 'noisy':
        actor = NoisyActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **kwargs)
    elif actor_type == 'random':
        actor = RandomActor(env.observation_space, env.action_space, goal_dim=goal_dim, **kwargs)
    elif actor_type == 'noised':
        actor = NoisedActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **kwargs)
    elif actor_type == 'slow':
        actor = SlowActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **kwargs)
    return actor

def evaluate_surrogate(cfg_input) -> dict:
    if not isinstance(cfg_input, DictConfig):
        cfg = OmegaConf.create(cfg_input)
    else:
        cfg = cfg_input

    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array' 
    }

    set_random_seed(cfg.seed)
    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(Actor(env.observation_space,env.action_space,goal_dim),model_path = cfg.expert_model, eval = True,device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=None,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )
    user_eval = evaluator.evaluate_raw_actor()
    print(f"User performance for actor {cfg.actor_type} with {cfg.actor_kwargs}: {user_eval}")

    return user_eval

if __name__ == "__main__":
    # actor_types = ['laggy', 'noisy']
    user_choice = input("Please choose exp (1 : Lander, 2 : Reacher): ")
    if user_choice.strip() == "1":
        exp_name = "Lander"
    elif user_choice.strip() == "2":
        exp_name = "Reacher"
    else:
        print("Unknown choice, use default: Lander")
        exp_name = "Lander"
    
    actor_types = ['laggy', 'noisy', 'noised', 'slow']
    # actor_types = ['slow']
    
    if exp_name == "Lander":
        expert_path = "${root_dir}/model_param/weight/lunar_expert/land_v1/best/LunarLander-v4__land_v1__42.actor.pth"
    elif exp_name == "Reacher":
        expert_path = "${root_dir}/model_param/weight/lunar_expert/reach_v1/best/LunarLander-v4__reach_v1__42.actor.pth"
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")
    
    
    actor_param_names = {
        'laggy': 'repeat_prob',
        'noisy': 'eps',
        'noised': 'noise_scale',
        'slow': 'speed'
    }

    
    hypervalues = [round(x * 0.05, 2) for x in range(21)]
    root_dir = '/code'
    seeds = [38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
    save_dir = os.path.join(root_dir, f'result_analysis/surrogate_performance/tables')
    os.makedirs(save_dir, exist_ok=True)
    # seeds = [38, 40]


    parameter_combinations = list(product(actor_types, hypervalues, seeds))
    print(f"Total experiments: {len(parameter_combinations)}")


    def run_experiment(params):
        import diffusha.envs.PegInsertion 
        actor_type, hyper_value, seed = params

        cfg = {
            'actor_type': actor_type,
            'actor_kwargs': {actor_param_names[actor_type]: hyper_value, 'seed': seed},
            'num_episode': 30,
            'root_dir': root_dir,
            'wandb_project_name': 'lunar_eval_surrogate',
            'wandb_log_dir': '${root_dir}/wandb_log/',
            'wandb_entity': None,
            'track': False,
            'save_video': False,
            'exp_name': exp_name,
            'env_id': 'LunarLander-v4' ,
            'task': 'land_v1' if exp_name == "Lander" else 'reach_v1',
            'seed': seed,  
            'max_episode_step': 1000,
            'device': 'cuda',
            'expert_model': expert_path,
            'random_initialization': False
        }

        result = evaluate_surrogate(cfg)
        result.update({
            'actor_type': actor_type,
            actor_param_names[actor_type]: hyper_value,
            'seed': seed
        })
        return result

    print("Starting parallel evaluations...")
    max_processes = 16
    with multiprocessing.Pool(processes=max_processes) as pool:
        results = pool.map(run_experiment, parameter_combinations)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, f'{exp_name}_eval_surrogate.csv'), index=False)
    print("All parameter sweeps completed.")

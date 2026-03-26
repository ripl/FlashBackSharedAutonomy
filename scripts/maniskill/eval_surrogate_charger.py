import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import multiprocessing
from itertools import product
import pandas as pd
from diffusha.model.ppo import ActorCritic_v3
from omegaconf import DictConfig, OmegaConf
from diffusha.actor.base import NoisyActor,LaggyActor,RandomActor,ExpertActor,NoisedActor,SlowActor

from diffusha.utils.eval_utils import Maniskill_Evaluator, write_to_csv
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
    from diffusha.utils.env_utils import set_random_seed
    import diffusha.envs.ChargerPlug
    if not isinstance(cfg_input, DictConfig):
        cfg = OmegaConf.create(cfg_input)
    else:
        cfg = cfg_input
    
    set_random_seed(cfg.seed)    
    env = gym.make(
    cfg.env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=cfg.n_env,
    obs_mode= cfg.obs_mode , # there is also "state_dict", "rgbd", ...
    control_mode=cfg.control_mode, # there is also "pd_joint_delta_pos", ...
    render_mode= cfg.render_mode,
    random_initial_position = True,
    simple_target_position = False,
    fix_size = True
    )

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True, device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)

    evaluator = Maniskill_Evaluator(
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
    actor_types = ['laggy', 'noisy', 'noised', 'slow']
    actor_param_names = {
        'laggy': 'repeat_prob',
        'noisy': 'eps',
        'noised': 'noise_scale',
        'slow': 'speed'
    }

    exp_name = "Charger"
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
            'wandb_project_name': 'maniskill_test',
            'wandb_log_dir': '${root_dir}/wandb_log/',
            'wandb_entity': None,
            'track': False,
            'save_video': False,
            'exp_name': "charger_surrogate",
            'env_id': 'SimpleChargerPlug-v1',
            'obs_mode': "state_dict",
            'control_mode': "pd_joint_delta_pos",
            'render_mode': "rgb_array",
            'n_env': 1,
            'seed': seed,  
            'max_episode_step': 300,
            'device': 'cuda',
            'expert_model': "${root_dir}/model_param/weight/charger_expert/continue_PPO_Curriculum_Charger_start0.05_end0.7/2025-04-01/19-18-09/CurriculumChargerPlug-v1__continue_PPO_Curriculum_Charger_start0.05_end0.7__4322.actor_critic.pth",
            'task': 'test_sweep',  
            'random_initialization': True
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

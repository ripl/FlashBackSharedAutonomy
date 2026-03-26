import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from diffusha.data_collection.generate_maniskill_data import DataGeneration 
from diffusha.actor.base import ExpertActor
from diffusha.model.ppo import ActorCritic_v3
from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig
from diffusha.model.model_utils import load_model
from diffusha import envs 
import gymnasium as gym 


@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'data_collection_maniskill_peg')
def collect_data(cfg: DictConfig) -> None:
    import diffusha.envs.PegInsertion 
    from diffusha.utils.env_utils import set_random_seed
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
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim)

    if cfg.load_model and cfg.model_path:
        expert_agent = load_model(expert_agent,cfg.model_path,eval=True,device=cfg.device)

    actor = ExpertActor(env.single_observation_space,env.single_observation_space,expert_agent,goal_dim)

    data_generator = DataGeneration(cfg.env_id,actor,cfg)
    data_generator.collect_data(env)
    env.close()

@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'data_collection_maniskill')
def collect_data_from_state_dicts(cfg: DictConfig) -> None:
    import diffusha.envs.PegInsertion 
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
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim)

    if cfg.load_model and cfg.model_path:
        expert_agent = load_model(expert_agent,cfg.model_path,eval=True,device=cfg.device)

    actor = ExpertActor(env.single_observation_space,env.single_observation_space,expert_agent,goal_dim)

    data_generator = DataGeneration(cfg.env_id,actor,cfg)
    data_generator.collect_from_state_dict(env,"data/debug")
    env.close()


if __name__ == '__main__':
    collect_data()

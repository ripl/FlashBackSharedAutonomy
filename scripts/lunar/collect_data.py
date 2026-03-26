import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from diffusha.data_collection.generate_data import DataGeneration 
from diffusha.actor.base import ExpertActor
from diffusha.model.sac import Actor
import hydra
from omegaconf import DictConfig
from diffusha.utils.env_utils import make_env
from diffusha.model.model_utils import load_model
from diffusha import envs 
from diffusha.utils.env_utils import set_random_seed


@hydra.main(version_base = None,
                  config_path = '../../configs/lunar',
                  config_name = 'data_collection')
def collect_data(cfg: DictConfig) -> None:
    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array' if cfg.save_video else 'human'
    }

    set_random_seed(cfg.seed)
    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = Actor(env.observation_space,env.action_space,goal_dim)

    if cfg.load_model and cfg.model_path:
        expert_agent = load_model(expert_agent,cfg.model_path,eval=True,device=cfg.device)

    actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim)
    data_generator = DataGeneration(cfg.env_id,actor,cfg)
    data_generator.collect_data(env)
    env.close()


if __name__ == '__main__':
    collect_data()

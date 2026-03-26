import os,sys 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch 
from diffusha.data_collection.generate_safetygym_data import DataGeneration 
from Safe_Policy_Optimization.safepo.common.env import make_sa_mujoco_env
from Safe_Policy_Optimization.safepo.common.model import ActorVCritic
from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig
import json 
import joblib
from diffusha.utils.env_utils import set_random_seed
from diffusha.envs.SafetyGymWrapper import DecomposeObservationWrapper
@hydra.main(version_base = None,
                  config_path = '../../configs/safetygym/',
                  config_name = 'data_collection_sa')
def collect_data(cfg: DictConfig) -> None:

    set_random_seed(cfg.seed)
    torch.set_num_threads(4)
    eval_dir = cfg.eval_dir
    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_id = config['task'] if 'task' in config.keys() else config['env_name']
    env_norms = os.listdir(eval_dir)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith('.pkl')]
    final_norm_name = sorted(env_norms)[-1]

    model_dir = eval_dir + '/torch_save'
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith('.pt')]
    final_model_name = sorted(models)[-1]

    model_path = model_dir + '/' + final_model_name
    norm_path = eval_dir + '/' + final_norm_name

    eval_env, obs_space, act_space = make_sa_mujoco_env(num_envs=1, env_id=env_id, seed=None)
    eval_env = DecomposeObservationWrapper(eval_env)
    model = ActorVCritic(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_sizes=config['hidden_sizes'],
        )
    model.actor.load_state_dict(torch.load(model_path))
    model.to(cfg.device)
    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
        eval_env.obs_rms = norm

    
    data_generator = DataGeneration(cfg.env_id,model,cfg)
    data_generator.collect_data(eval_env)
    eval_env.close()


if __name__ == '__main__':
    collect_data()

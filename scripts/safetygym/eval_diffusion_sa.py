import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch 
from Safe_Policy_Optimization.safepo.common.env import make_sa_mujoco_env
from Safe_Policy_Optimization.safepo.common.model import ActorVCritic
from diffusha.actor.assistive_actor import DiffusionAssistedActor
from diffusha.model.model_utils import load_model,load_transform
from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig,OmegaConf
import json 
import joblib
from diffusha.utils.env_utils import set_random_seed
from diffusha.envs.SafetyGymWrapper import DecomposeObservationWrapper
from diffusha.actor.base import NoisyActor,LaggyActor,RandomActor,ExpertActor,NoisedActor,SafetyGymActorWrapper
from diffusha.utils.eval_utils import SafetyGym_Evaluator,write_to_csv
from diffusha.algo.algo_utils import Transform,load_cm_models_path
from diffusha.algo.diffusion_policy import JointConditionPolicy

def build_actor(type,env,expert_actor,goal_dim,**kwargs):
    assert type in ['laggy','noisy','random','noised']
    if type == 'laggy':
        actor = LaggyActor(env.observation_space,env.action_space,expert_actor,goal_dim=goal_dim,**kwargs)
    elif type == 'noisy':
        actor = NoisyActor(env.observation_space,env.action_space,expert_actor,goal_dim=goal_dim,**kwargs)
    elif type == 'random':
        actor = RandomActor(env.observation_space,env.action_space,goal_dim=goal_dim,**kwargs)
    elif type == 'noised':
        actor = NoisedActor(env.observation_space,env.action_space,expert_actor,goal_dim=goal_dim,**kwargs)
    return actor 

@hydra.main(version_base = None,
                  config_path = '../../configs/safetygym/',
                  config_name = 'eval_diffusion_sa')
def eval_safegym(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    torch.set_num_threads(4)

    #####  Load Expert and Env 
    eval_dir = cfg.expert_dir
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
    expert_agent = SafetyGymActorWrapper(model)
    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
        eval_env.obs_rms = norm
    
    ## Expert Policy 
    goal_dim = eval_env.goal_dim
    expert_actor = ExpertActor(eval_env.observation_space,eval_env.action_space,expert_agent,goal_dim,cfg.device)
    
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,eval_env,expert_actor,goal_dim,**cfg.actor_kwargs)

    ## Policy Part 2
    denoiser = hydra.utils.instantiate(cfg.joint_cm_denoiser)
    joint_cm_model = hydra.utils.instantiate(cfg.joint_cm_model)
    fwd_cfg_path = os.path.join(hydra.utils.get_original_cwd(),cfg.fwd_trainer_cfg_path)            
    fwd_cfg = OmegaConf.load(fwd_cfg_path)
    fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
    joint_cm_transform = Transform(mean = 0.5, std = 0.5,device=cfg.device)
    
    matched_paths = load_cm_models_path(cfg.model_dir)
    joint_cm_model = load_model(joint_cm_model,matched_paths['cm_model'], eval = True,device=cfg.device)
    fwd_model = load_model(fwd_model,matched_paths['fwd_model'], eval = True,device=cfg.device)
    joint_cm_transform.load_stats(matched_paths['transform_stats'])
    
    joint_cm_policy = JointConditionPolicy(denoiser,
                                            joint_cm_model,
                                            fwd_model,
                                            joint_cm_transform,
                                            **cfg.condition_policy)
    

    ## Policy Part 3
    assistive_policy = DiffusionAssistedActor(eval_env.observation_space,eval_env.action_space,joint_cm_policy,user_policy)

    evaluator = SafetyGym_Evaluator(
        cfg,eval_env,raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )
    if cfg.eval_user: 
        user_eval = evaluator.evaluate_raw_actor()
        print(f"User performance: {user_eval}")
        write_to_csv(cfg.result_save_dir,user_eval,cfg,method_name="User")
    else:
        assistive_eval = evaluator.evalute_assitive_actor()
        print(f"Assistive performance: {assistive_eval}")
        write_to_csv(cfg.result_save_dir,assistive_eval,cfg)
   

if __name__ == '__main__':
    eval_safegym()

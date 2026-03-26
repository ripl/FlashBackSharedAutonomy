import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from diffusha.utils.eval_utils import Evaluator,write_to_csv
from diffusha.algo.diffusion_policy import EDMPolicy,CMPolicy,JointConditionPolicy,DDPMPolicy
from diffusha.actor.base import NoisyActor,LaggyActor,RandomActor,ExpertActor,NoisedActor, SlowActor
from diffusha.model.sac import Actor
from diffusha.actor.assistive_actor import DiffusionAssistedActor
from diffusha.model.model_utils import load_model,load_transform
from diffusha import envs
import hydra
from omegaconf import DictConfig,OmegaConf
from diffusha.utils.env_utils import make_env
import gymnasium as gym 
import torch as th
from diffusha.utils.env_utils import set_random_seed
from diffusha.algo.algo_utils import Transform,load_cm_models_path


# def build_actor(type,env,expert_actor,goal_dim,**kwargs):
#     assert type in ['laggy','noisy','random','noised', 'slow']
#     if type == 'laggy':
#         actor = LaggyActor(env.observation_space,env.action_space,expert_actor,goal_dim=goal_dim,**kwargs)
#     elif type == 'noisy':
#         actor = NoisyActor(env.observation_space,env.action_space,expert_actor,goal_dim=goal_dim,**kwargs)
#     elif type == 'random':
#         actor = RandomActor(env.observation_space,env.action_space,goal_dim=goal_dim,**kwargs)
#     elif type == 'noised':
#         actor = NoisedActor(env.observation_space,env.action_space,expert_actor,goal_dim=goal_dim,**kwargs)
#     elif type == 'slow':
#         actor = SlowActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **kwargs)
#     return actor 

def build_actor(actor_type, env, expert_actor, goal_dim, **kwargs):
    valid_keys = {
        "noisy": {"eps", "preserve_norm", "seed"},
        "laggy": {"repeat_prob", "seed"},
        "noised": {"noise_scale", "preserve_norm", "seed"},
        "slow": {"speed", "seed"}
    }
    allowed = valid_keys.get(actor_type, set())
    # filter out unsupported kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    
    if actor_type == "laggy":
        return LaggyActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **filtered_kwargs)
    elif actor_type == "slow":
        return SlowActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **filtered_kwargs)
    elif actor_type == "noisy":
        return NoisyActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **filtered_kwargs)
    elif actor_type == "noised":
        return NoisedActor(env.observation_space, env.action_space, expert_actor, goal_dim=goal_dim, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown actor type: {actor_type}")



@hydra.main(config_path = '../../configs/lunar',
            config_name = 'eval_joint_edm',version_base=None)
def evaluate_joint_edm(cfg: DictConfig) -> None:
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
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)


    ## Policy Part 2
    denoiser = hydra.utils.instantiate(cfg.joint_edm_denoiser)
    joint_edm_model = hydra.utils.instantiate(cfg.joint_edm_model)
    joint_edm_model = load_model(joint_edm_model,cfg.joint_edm_model_path, eval = True,device=cfg.device)

    from diffusha.algo.expert_trainer import ForwardTrainer
    fwd_cfg_path = os.path.join(hydra.utils.get_original_cwd(),cfg.fwd_trainer_cfg_path)            
    fwd_cfg = OmegaConf.load(fwd_cfg_path)
    fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
    fwd_model = load_model(fwd_model,cfg.fwd_model_path, eval = True,device=cfg.device)

    joint_edm_transform = load_transform(cfg.joint_edm_transform_path)
    
    joint_edm_policy = JointConditionPolicy(denoiser,
                                            joint_edm_model,
                                            fwd_model,
                                            joint_edm_transform,
                                            **cfg.condition_policy)

    ## Policy Part 3
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,joint_edm_policy,user_policy)

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
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

@hydra.main(config_path = '../../configs/lunar',
            config_name = 'eval_joint_cm',version_base=None)
def evaluate_joint_cm(cfg):
    # cfg.hydra.run.dir = '/tmp/hydra_outputs'
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
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)
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
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,joint_cm_policy,user_policy)

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
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
        if cfg.noise_level == cfg.n_steps: # assume no assistance
            assistive_eval = evaluator.evaluate_raw_actor()
        else:
            assistive_eval = evaluator.evalute_assitive_actor()
        print(f"Assistive performance: {assistive_eval}")
        write_to_csv(cfg.result_save_dir,assistive_eval,cfg)


@hydra.main(config_path = '../../configs/lunar',
            config_name = 'eval_joint_cm_blend',version_base=None)
def evaluate_joint_cm_blend(cfg):
    # cfg.hydra.run.dir = '/tmp/hydra_outputs'
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
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)
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
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,joint_cm_policy,user_policy)

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )

    assistive_eval = evaluator.evaluate_blend_actor()
    print(f"Blend performance: {assistive_eval}")
    write_to_csv(cfg.result_save_dir,assistive_eval,cfg)

@hydra.main(config_path = '../../configs/lunar',
            config_name = 'eval_ddpm',version_base=None)
def evaluate_ddpm(cfg: DictConfig) -> None:
    # env = gym.make(cfg.env_id,continuous = True)
    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array'
    }
    from colorama import Fore

    print(Fore.YELLOW + "Running Configs: " + Fore.RESET,cfg)
    set_random_seed(cfg.seed)
    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(Actor(env.observation_space,env.action_space,goal_dim),model_path = cfg.expert_model, eval = True,device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)


    ## Policy Part 2
    ddpm_model = hydra.utils.instantiate(cfg.ddpm_model)
    ddpm_model = load_model(ddpm_model,cfg.ddpm_model_path, eval = True,device=cfg.device)
    ddpm_denoiser = hydra.utils.instantiate(cfg.ddpm_denoiser, model=ddpm_model)

    ddpm_policy = hydra.utils.instantiate(cfg.ddpm_policy, denoiser=ddpm_denoiser)

    ## Policy Part 3
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,ddpm_policy,user_policy)

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )
    if cfg.eval_user: 
        user_eval = evaluator.evaluate_raw_actor()
        print(f"User performance: {user_eval}")
        write_to_csv(cfg.result_save_dir,user_eval,cfg,method_name="User", is_ddpm=True)
    else:
        assistive_eval = evaluator.evalute_assitive_actor()
        print(f"Assistive performance: {assistive_eval}")
        write_to_csv(cfg.result_save_dir,assistive_eval,cfg, is_ddpm=True)



if __name__ == '__main__':
    # evaluate_ddpm()
    # evaluate_joint_cm()
    evaluate_joint_cm_blend()


    
import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from diffusha.utils.eval_utils import Maniskill_Evaluator, write_to_csv
from diffusha.algo.diffusion_policy import EDMPolicy,CMPolicy,JointConditionPolicy,DDPMPolicy
from diffusha.actor.base import NoisyActor,LaggyActor,RandomActor,ExpertActor,NoisedActor, SlowActor
from diffusha.model.ppo import ActorCritic_v3
from diffusha.actor.assistive_actor import DiffusionAssistedActor
from diffusha.model.model_utils import load_model
from diffusha import envs
import hydra
from omegaconf import DictConfig,OmegaConf
from diffusha.utils.env_utils import make_env
import gymnasium as gym 
import torch as th
from diffusha.utils.env_utils import set_random_seed
from diffusha.algo.algo_utils import Transform,load_cm_models_path


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


@hydra.main(config_path = '../../configs/maniskill/',
            config_name = 'eval_diffusion_peg_cm',version_base=None)
            # config_name = 'eval_diffusion_charger',version_base=None)
def evaluate_cm_peg(cfg: DictConfig) -> None:
    import diffusha.envs.PegInsertion
    set_random_seed(cfg.seed)
    env = gym.make(
    cfg.env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=cfg.n_env,
    obs_mode= cfg.obs_mode , # there is also "state_dict", "rgbd", ...
    control_mode=cfg.control_mode, # there is also "pd_joint_delta_pos", ...
    render_mode= cfg.render_mode,
    random_initial_position = True,
    simple_target_position = False,
    fix_size = True)

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True,device = cfg.device)
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
    # breakpoint()
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

    evaluator = Maniskill_Evaluator(
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


@hydra.main(config_path = '../../configs/maniskill/',
            config_name = 'eval_diffusion_peg_cm_blend',version_base=None)
def evaluate_cm_peg_blend(cfg: DictConfig) -> None:
    import diffusha.envs.PegInsertion
    set_random_seed(cfg.seed)
    env = gym.make(
    cfg.env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=cfg.n_env,
    obs_mode= cfg.obs_mode , # there is also "state_dict", "rgbd", ...
    control_mode=cfg.control_mode, # there is also "pd_joint_delta_pos", ...
    render_mode= cfg.render_mode,
    random_initial_position = True,
    simple_target_position = False,
    fix_size = True)

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True,device = cfg.device)
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
    # breakpoint()
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

    evaluator = Maniskill_Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )

    user_eval = evaluator.evaluate_blend_actor()
    print(f"Blend performance: {user_eval}")
    write_to_csv(cfg.result_save_dir,user_eval,cfg,method_name="Blend")



@hydra.main(config_path = '../../configs/maniskill/',
            config_name = 'eval_diffusion_charger_cm',version_base=None)
def evaluate_cm_charger(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    import diffusha.envs.ChargerPlug

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
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True,device = cfg.device)
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
    # breakpoint()
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

    evaluator = Maniskill_Evaluator(
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


@hydra.main(config_path = '../../configs/maniskill/',
            config_name = 'eval_diffusion_charger_cm_blend',version_base=None)
def evaluate_cm_charger_blend(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    import diffusha.envs.ChargerPlug

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
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True,device = cfg.device)
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
    # breakpoint()
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

    evaluator = Maniskill_Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )

    user_eval = evaluator.evaluate_blend_actor()
    print(f"Blend performance: {user_eval}")
    write_to_csv(cfg.result_save_dir,user_eval,cfg,method_name="Blend")



@hydra.main(config_path = '../../configs/maniskill/',
            config_name = 'eval_diffusion_peg_ddpm',version_base=None)
            # config_name = 'eval_diffusion_charger',version_base=None)
def evaluate_ddpm_peg(cfg: DictConfig) -> None:
    import diffusha.envs.PegInsertion
    set_random_seed(cfg.seed)
    env = gym.make(
    cfg.env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=cfg.n_env,
    obs_mode= cfg.obs_mode , # there is also "state_dict", "rgbd", ...
    control_mode=cfg.control_mode, # there is also "pd_joint_delta_pos", ...
    render_mode= cfg.render_mode,
    random_initial_position = True, # False here
    simple_target_position = False,
    fix_size = True)

    ## Expert Policy 
    # breakpoint()
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True,device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)


    ## Policy Part 2
    ddpm_model = hydra.utils.instantiate(cfg.ddpm_model)
    ddpm_model = load_model(ddpm_model,cfg.ddpm_model_path, eval = True,device=cfg.device)
    ddpm_denoiser = hydra.utils.instantiate(cfg.ddpm_denoiser, model=ddpm_model, device = cfg.device)

    ddpm_policy = hydra.utils.instantiate(cfg.ddpm_policy, denoiser=ddpm_denoiser)
    

    ## Policy Part 3
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,ddpm_policy,user_policy)

    evaluator = Maniskill_Evaluator(
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
        if cfg.ddpm_policy.fwd_diffuse_ratio == 0.0:
            assistive_eval = evaluator.evaluate_raw_actor()
        else:
            assistive_eval = evaluator.evalute_assitive_actor()
        print(f"Assistive performance: {assistive_eval}")
        write_to_csv(cfg.result_save_dir,assistive_eval,cfg, is_ddpm=True)


@hydra.main(config_path = '../../configs/maniskill/',
            config_name = 'eval_diffusion_charger_ddpm',version_base=None)
            # config_name = 'eval_diffusion_charger',version_base=None)
def evaluate_ddpm_charger(cfg: DictConfig) -> None:
    import json
    set_random_seed(cfg.seed)
    import diffusha.envs.ChargerPlug
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
    # breakpoint()
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(ActorCritic_v3(env.single_observation_space,env.single_action_space,goal_dim,action_rescale=0.3),model_path = cfg.expert_model, eval = True,device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)


    ## Policy Part 2
    ddpm_model = hydra.utils.instantiate(cfg.ddpm_model)
    ddpm_model = load_model(ddpm_model,cfg.ddpm_model_path, eval = True,device=cfg.device)
    ddpm_denoiser = hydra.utils.instantiate(cfg.ddpm_denoiser, model=ddpm_model, device = cfg.device)

    ddpm_policy = hydra.utils.instantiate(cfg.ddpm_policy, denoiser=ddpm_denoiser)
    

    ## Policy Part 3
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,ddpm_policy,user_policy)

    evaluator = Maniskill_Evaluator(
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
    # evaluate_ddpm_peg()
    # evaluate_ddpm_charger()
    # evaluate_cm_peg() 
    # evaluate_cm_charger()

    # evaluate_cm_peg_blend()
    evaluate_cm_charger_blend()

    
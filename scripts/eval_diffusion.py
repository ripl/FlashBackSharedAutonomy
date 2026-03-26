import os,sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from diffusha.utils.eval_utils import Evaluator,write_to_csv
from diffusha.algo.diffusion_policy import EDMPolicy,CMPolicy,JointConditionPolicy,DDPMPolicy
from diffusha.actor.base import NoisyActor,LaggyActor,RandomActor,ExpertActor,NoisedActor
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
from diffusha.algo.algo_utils import Transform, load_cm_models_path


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



@hydra.main(config_path = '../configs/deprecated',
            config_name = 'eval_edm',version_base=None)
def evaluate_edm(cfg: DictConfig) -> None:
    # env = gym.make(cfg.env_id,continuous = True)
    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array' 
    }
    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(Actor(env.observation_space,env.action_space,goal_dim),model_path = cfg.expert_model, eval = True,device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)


    ## Policy Part 2
    denoiser = hydra.utils.instantiate(cfg.denoiser)
    edm_model = hydra.utils.instantiate(cfg.edm_model)
    edm_model = load_model(edm_model,cfg.edm_model_path, eval = True,device=cfg.device)
    
    edm_policy = EDMPolicy(denoiser,edm_model,**cfg.edm_policy)

    ## Policy Part 3
    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,edm_policy,user_policy)

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )
    user_eval = evaluator.evaluate_raw_actor()
    assistive_eval = evaluator.evalute_assitive_actor()
    print(f"User performance: {user_eval}")
    print(f"Assistive performance: {assistive_eval}")

@hydra.main(config_path = '../configs/deprecated',
            config_name = 'eval_cm',version_base=None)
def evaluate_cm(cfg: DictConfig) -> None:
    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array' 
    }
    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()

    ## Expert Policy 
    goal_dim = env.unwrapped.goal_dim if hasattr(env.unwrapped, "goal_dim") else 0
    expert_agent = load_model(Actor(env.observation_space,env.action_space,goal_dim),model_path = cfg.expert_model, eval = True,device = cfg.device)
    expert_actor = ExpertActor(env.observation_space,env.action_space,expert_agent,goal_dim,cfg.device)
    ## Policy Part 1
    
    user_policy = build_actor(cfg.actor_type,env,expert_actor,goal_dim,**cfg.actor_kwargs)

    ##
    denoiser = hydra.utils.instantiate(cfg.denoiser)
    cm_model = hydra.utils.instantiate(cfg.cm_model)
    cm_model = load_model(cm_model,cfg.cm_model_path, eval = True,device=cfg.device)
    cm_transform = load_transform(cfg.cm_transform_path)

    cm_policy = CMPolicy(denoiser,cm_model,cm_transform,**cfg.cm_policy)

    assistive_policy = DiffusionAssistedActor(env.observation_space,env.action_space,cm_policy,user_policy)
    
    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device = cfg.device
    )
    user_eval = evaluator.evaluate_raw_actor()
    assistive_eval = evaluator.evalute_assitive_actor()
    print(f"User performance: {user_eval}")
    print(f"Assistive performance: {assistive_eval}")

@hydra.main(config_path = '../configs/lunar',
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
    user_eval = evaluator.evaluate_raw_actor()
    assistive_eval = evaluator.evalute_assitive_actor()
    print(f"User performance: {user_eval}")
    print(f"Assistive performance: {assistive_eval}")
    rs = getattr(cfg, "result_save_dir", None)
    if rs:
        write_to_csv(rs, assistive_eval, cfg)


@hydra.main(config_path = '../configs/lunar',
            config_name = 'eval_joint_cm',version_base=None)
def evaluate_joint_cm(cfg):
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
    fwd_cfg_path = os.path.join(hydra.utils.get_original_cwd(), cfg.fwd_trainer_cfg_path)
    fwd_cfg = OmegaConf.load(fwd_cfg_path)
    fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
    joint_cm_transform = Transform(mean=0.5, std=0.5, device=cfg.device)

    matched_paths = load_cm_models_path(cfg.model_dir)
    joint_cm_model = load_model(joint_cm_model, matched_paths["cm_model"], eval=True, device=cfg.device)
    fwd_model = load_model(fwd_model, matched_paths["fwd_model"], eval=True, device=cfg.device)
    joint_cm_transform.load_stats(matched_paths["transform_stats"])

    joint_cm_policy = JointConditionPolicy(
        denoiser,
        joint_cm_model,
        fwd_model,
        joint_cm_transform,
        **cfg.condition_policy,
    )

    assistive_policy = DiffusionAssistedActor(
        env.observation_space, env.action_space, joint_cm_policy, user_policy
    )

    evaluator = Evaluator(
        cfg,
        env,
        raw_actor=user_policy,
        assistive_actor=assistive_policy,
        num_episode=cfg.num_episode,
        max_episode_step=cfg.max_episode_step,
        device=cfg.device,
    )
    if cfg.eval_user:
        user_eval = evaluator.evaluate_raw_actor()
        print(f"User performance: {user_eval}")
        write_to_csv(cfg.result_save_dir, user_eval, cfg, method_name="User")
    else:
        if cfg.noise_level == cfg.n_steps:
            assistive_eval = evaluator.evaluate_raw_actor()
        else:
            assistive_eval = evaluator.evalute_assitive_actor()
        print(f"Assistive performance: {assistive_eval}")
        write_to_csv(cfg.result_save_dir, assistive_eval, cfg)

@hydra.main(config_path = '../configs/lunar',
            config_name = 'eval_ddpm',version_base=None)
def evaluate_ddpm(cfg: DictConfig) -> None:
    import json
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
    ddpm_denoiser = hydra.utils.instantiate(cfg.ddpm_denoiser, model=ddpm_model, device=cfg.device)

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

    output_dir = "ddpm_results"
    os.makedirs(output_dir, exist_ok=True)
    if "sweep" in cfg.suffix:
        os.makedirs(os.path.join(output_dir,f"eval_{cfg.task}_beta_{cfg.ddpm_denoiser.beta_schedule}_{cfg.ddpm_denoiser.beta_max}_{cfg.actor_type}_{cfg.suffix}"), exist_ok=True)
        output_file = os.path.join(output_dir, f"eval_{cfg.task}_beta_{cfg.ddpm_denoiser.beta_schedule}_{cfg.ddpm_denoiser.beta_max}_{cfg.actor_type}_{cfg.suffix}/seed_{cfg.seed}.json")
    elif "expert" in cfg.suffix:
        output_file = os.path.join(output_dir, f"eval_{cfg.task}_expert.json")
    elif cfg.suffix and "sweep" not in cfg.suffix:
        output_file = os.path.join(output_dir, f"eval_{cfg.task}_seed_{cfg.seed}_beta_{cfg.ddpm_denoiser.beta_schedule}_{cfg.ddpm_denoiser.beta_max}_{cfg.actor_type}_{cfg.suffix}.json")
    else:
        output_file = os.path.join(output_dir, f"eval_{cfg.task}_seed_{cfg.seed}_beta_{cfg.ddpm_denoiser.beta_schedule}_{cfg.ddpm_denoiser.beta_max}_{cfg.actor_type}.json")


    # Only run user evaluation for the first time to shorten the evaluation time
    if "expert" in cfg.suffix:
        print(f"Evaluate expert policy")
        user_eval = evaluator.evaluate_raw_actor()
        assistive_eval = None
    elif not os.path.exists(output_file):
        # Run user evaluation
        print(f"Output file does not exist. Running user evaluation and assistive evaluation.")
        user_eval = evaluator.evaluate_raw_actor()
        assistive_eval = evaluator.evalute_assitive_actor()
    else:
        # Only run assistive evaluation
        print(f"Output file exists. Only running assistive evaluation.")
        user_eval = None  # Indicating no user evaluation is performed
        assistive_eval = evaluator.evalute_assitive_actor()

        # Save results to a JSON file
    result = {
        "fwd_diffuse_ratio": cfg.ddpm_policy.fwd_diffuse_ratio,
        "user_performance": user_eval,
        "assistive_performance": assistive_eval,
        "seed": cfg.seed,
        "task": cfg.task,
        "beta_min": cfg.ddpm_denoiser.beta_min,
        "beta_max": cfg.ddpm_denoiser.beta_max,
        "beta_schedule": cfg.ddpm_denoiser.beta_schedule,
        "actor": cfg.actor_type,
        "flawed_eps": cfg.actor_kwargs.eps,
        "random_init": cfg.random_initialization,
    }

    if cfg.save_json:
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                try:
                    data = json.load(f) 
                    if not isinstance(data, list):  
                        data = []
                except json.JSONDecodeError:
                    data = []  
        else:
            data = []

        data.append(result)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

    print(f"User performance: {user_eval}")
    print(f"Assistive performance: {assistive_eval}")
    print(f"Results saved to {output_file}")



if __name__ == '__main__':
    # evaluate_edm()
    # evaluate_cm()
    # evaluate_ddpm()
    # evaluate_cm()
    evaluate_ddpm()
    # evaluate_joint_edm()
    # evaluate_joint_cm()


    
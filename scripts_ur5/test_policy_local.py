import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

@hydra.main(config_path = '../configs_ur5',
            config_name = 'eval_ddpm',version_base=None)
def evaluate_ddpm(cfg: DictConfig) -> None:

    ddpm_model = hydra.utils.instantiate(cfg.ddpm_model)
    ddpm_model = load_model(ddpm_model,cfg.ddpm_model_path, eval = True,device=cfg.device)
    ddpm_denoiser = hydra.utils.instantiate(cfg.ddpm_denoiser, model=ddpm_model)

    ddpm_policy = hydra.utils.instantiate(cfg.ddpm_policy, denoiser=ddpm_denoiser)

    import time
    import random

    while True:
        random_action = [round(random.uniform(0.2, 0.8), 3) for _ in range(3)]
        random_obs = [round(random.uniform(0.2, 0.8), 3) for _ in range(9)]
        print("random_action: ", random_action)    
        shared_action = ddpm_policy.act(random_obs, random_action)
        print("shared_action: ", shared_action)
        time.sleep(1)



if __name__ == '__main__':
    evaluate_ddpm()
    # evaluate_joint_cm()


    
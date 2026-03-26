import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import hydra
from omegaconf import DictConfig
from diffusha.utils.env_utils import make_env
import gymnasium as gym 
from diffusha.utils.env_utils import set_random_seed


@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'expert_ppo_charger')
def train_charger_expert(cfg: DictConfig) -> None:
    from diffusha.algo.maniskill_ppo_trainer import ExpertTrainer
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

    expert_trainer = ExpertTrainer(env, cfg)
    if cfg.load_dir is not None:
        expert_trainer.load(cfg.load_dir)
    expert_trainer.run(env,forward_trainer=None)
    expert_trainer.save()


@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'expert_ppo_cl_charger')
def train_CL_charger_expert(cfg: DictConfig) -> None:
    from diffusha.algo.maniskill_ppo_trainer import ExpertTrainer
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

    expert_trainer = ExpertTrainer(env, cfg)
    if cfg.load_dir is not None:
        expert_trainer.load(cfg.load_dir)
    expert_trainer.CL_run(env,forward_trainer=None)
    expert_trainer.save()

@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'expert_bc')
def train_bc_expert(cfg:DictConfig):
    from diffusha.algo.maniskill_bc import BCTrainer
    set_random_seed(cfg.seed)
    eval_env = gym.make(
    cfg.env_id, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode= cfg.obs_mode , # there is also "state_dict", "rgbd", ...
    control_mode=cfg.control_mode, # there is also "pd_joint_delta_pos", ...
    render_mode= cfg.render_mode,
    random_initial_position = False,
    simple_target_position = True,
    fix_size = True
    )
    bc_trainer = BCTrainer(eval_env,cfg)
    bc_trainer.prepare_data(cfg.data_dir,add_noise=cfg.add_noise)
    bc_trainer.run(n_iters=cfg.total_iterations, eval_env=eval_env)
    bc_trainer.save()


if __name__ == '__main__':
    # train_bc_expert()
    train_charger_expert()
    # train_CL_charger_expert()

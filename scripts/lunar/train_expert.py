import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from diffusha.algo.expert_trainer import ExpertTrainer,ForwardTrainer
from diffusha.algo.old_expert_trainer import ExpertTrainer as OldExpertTrainer 
import hydra
from omegaconf import DictConfig,OmegaConf
from diffusha.utils.env_utils import make_env
import gymnasium as gym 
from diffusha import envs 
from diffusha.utils.env_utils import set_random_seed

# For: Sweep fixing the issue with hydra and argparse
import sys
new_args = [sys.argv[0]]
for arg in sys.argv[1:]:
    if arg.startswith('--') and '=' in arg:
        new_args.append(arg[2:])
    else:
        new_args.append(arg)
sys.argv = new_args

@hydra.main(version_base = None,
                  config_path = '../../configs/lunar',
                  config_name = 'expert')

def train_expert(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array' 
    }
    ### Fix Random Seet 
    
    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()

    # if cfg.use_forward_model:
    #     fwd_cfg = OmegaConf.load(cfg.fwd_trainer_cfg_path)
    #     fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
    #     fwd_trainer = ForwardTrainer(fwd_model,fwd_cfg.forward_trainer)

    expert_trainer = ExpertTrainer(env, cfg)

    # if cfg.load_dir is not None:
    #     expert_trainer.load(cfg.load_dir)
    expert_trainer.run(env)
    # expert_trainer.run(env,forward_trainer=fwd_trainer)
    expert_trainer.save()
    # if cfg.use_forward_model:
        # fwd_trainer.save()

@hydra.main(version_base = None,
                  config_path = '../../configs/lunar',
                  config_name = 'expert')

def sweep_train_expert(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    if cfg.track:
        import wandb
        # init wandb and use TensorBoard sync
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            save_code=True,
            name=f"{cfg.env_id}__{cfg.exp_name}__{cfg.task}__{cfg.seed}"
        )
        cfg = wandb.config
    
    args_dict = vars(cfg)
    for key, value in args_dict.items():
        try:
            wandb.config[key] = value
        except Exception as e:
            print(f"Failed to log {key}: {value} ({type(value)}) - {e}")

    env_kwargs = {
        "continuous": True,
        "task":cfg.task,
        "random_initialization": cfg.random_initialization,
        "max_steps":1000,
        'render_mode': 'rgb_array' 
    }

    env = make_env(cfg.env_id, cfg.seed, 0, False, cfg.exp_name, **env_kwargs)()

    # if cfg.use_forward_model:
    #     fwd_cfg = OmegaConf.load(cfg.fwd_trainer_cfg_path)
    #     fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
    #     fwd_trainer = ForwardTrainer(fwd_model,fwd_cfg.forward_trainer)

    expert_trainer = ExpertTrainer(env, cfg)
    # expert_trainer = OldExpertTrainer(env, cfg)

    expert_trainer.run(env)
    # expert_trainer.run(env,forward_trainer=fwd_trainer)
    expert_trainer.save()
    # if cfg.use_forward_model:
    #     fwd_trainer.save()


if __name__ == '__main__':
    sweep_train_expert()
import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import hydra
from omegaconf import DictConfig,OmegaConf

from diffusha import envs 
from diffusha.utils.env_utils import set_random_seed
from copy import deepcopy 

    
@hydra.main(config_path = '../../configs/safetygym/',
            config_name = 'train_joint_edm_sa.yaml')
def train_joint_edm(cfg: DictConfig) -> None:
    from diffusha.algo.joint_trainer import JointEDMTrainer

    set_random_seed(cfg.seed)
    joint_edm_model = hydra.utils.instantiate(cfg.joint_edm_model)
    joint_denosier = hydra.utils.instantiate(cfg.joint_edm_denoiser)
    
    dataset = hydra.utils.instantiate(cfg.dataset)

    if cfg.use_weighting_model:
        from diffusha.model.diffusion_model import AdaptiveWeighting
        weighting_model = AdaptiveWeighting(cfg.joint_edm_model.hidden_dim)
    else:
        weighting_model = None
    joint_trainer = JointEDMTrainer(joint_edm_model, joint_denosier, weighting_model=weighting_model, args = cfg)

    if cfg.use_forward_model:
        from diffusha.algo.expert_trainer import ForwardTrainer
        fwd_cfg_path = os.path.join(hydra.utils.get_original_cwd(),cfg.fwd_trainer_cfg_path)            
        fwd_cfg = OmegaConf.load(fwd_cfg_path)
        fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
        fwd_cfg.forward_trainer.exp_name = cfg.exp_name
        fwd_cfg.forward_trainer.save_dir = cfg.save_dir
        fwd_trainer = ForwardTrainer(fwd_model,fwd_cfg.forward_trainer)
        if cfg.load_fwd_path is not None:
            fwd_trainer.load(cfg.load_fwd_path)
    else:
        fwd_trainer = None

    joint_trainer.run_loop(cfg.n_iter, 
                         dataset, 
                         cfg.batch_size, 
                         cfg.gamma,
                         forward_trainer=fwd_trainer)
    joint_trainer.save()
    if cfg.use_forward_model:
        fwd_trainer.save()

@hydra.main(config_path = '../../configs/safetygym/',
            config_name = 'train_joint_cm_sa.yaml')
def train_joint_cm(cfg:DictConfig):
    from diffusha.algo.joint_trainer import JointCMTrainer
    set_random_seed(cfg.seed)
    joint_cm_model = hydra.utils.instantiate(cfg.joint_cm_model)
    target_model = deepcopy(joint_cm_model)
    joint_denosier = hydra.utils.instantiate(cfg.joint_cm_denoiser)
    
    teacher_model = hydra.utils.instantiate(cfg.teacher_model)
    teacher_denoiser = hydra.utils.instantiate(cfg.teacher_denoiser)

    if cfg.use_weighting_model:
        from diffusha.model.diffusion_model import AdaptiveWeighting
        weighting_model = AdaptiveWeighting(cfg.joint_cm_model.hidden_dim)
    else:
        weighting_model = None
    joint_trainer = JointCMTrainer(joint_cm_model, 
                                joint_denosier,
                                target_model, 
                                teacher_model,
                                teacher_denoiser, 
                                weighting_model=weighting_model,
                                args = cfg)
    
    dataset = hydra.utils.instantiate(cfg.dataset)
    if cfg.use_forward_model:
        from diffusha.algo.expert_trainer import ForwardTrainer
        fwd_cfg_path = os.path.join(hydra.utils.get_original_cwd(),cfg.fwd_trainer_cfg_path)            
        fwd_cfg = OmegaConf.load(fwd_cfg_path)
        fwd_cfg.forward_trainer.exp_name = cfg.exp_name
        fwd_cfg.forward_trainer.save_dir = cfg.save_dir
        fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
        fwd_trainer = ForwardTrainer(fwd_model,fwd_cfg.forward_trainer)
    else:
        fwd_trainer = None
    
    joint_trainer.run_loop(cfg.n_iter, 
                        dataset, 
                        cfg.batch_size,
                        cfg.gamma,
                        fwd_trainer)
    joint_trainer.save()
    if cfg.use_forward_model:
        fwd_trainer.save()

if __name__ == '__main__':
    # train_joint_edm()
    train_joint_cm()
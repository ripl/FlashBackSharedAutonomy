import os,sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from diffusha.algo.edm_trainer import EDMTrainer 
from diffusha.algo.cm_trainer import CMTrainer
from diffusha.algo.ddpm_trainer import DDPMTrainer
from diffusha.data_collection.buffer import ExpertTransitionDataset
from hydra.core.hydra_config import OmegaConf
import hydra
from omegaconf import DictConfig
from diffusha.utils.env_utils import make_env
from copy import deepcopy
from diffusha.utils.env_utils import set_random_seed
from diffusha.model.model_utils import load_model,load_transform


@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'train_ddpm_charger')
def train_ddpm(cfg: DictConfig) -> None:

    set_random_seed(cfg.seed)
    ddpm_model = hydra.utils.instantiate(cfg.ddpm_model)
    target_model = deepcopy(ddpm_model)

    denoiser = hydra.utils.instantiate(cfg.ddpm_denoiser, model=ddpm_model)
    
    ddpm_trainer = DDPMTrainer(denoiser, cfg)
    
    dataset = hydra.utils.instantiate(cfg.dataset)
    
    ddpm_trainer.run_loop(cfg.n_iter, 
                        dataset, 
                        cfg.batch_size)
    ddpm_trainer.save()
    
@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'train_joint_edm_peg')
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

@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'train_joint_cm_peg')
def train_joint_cm(cfg:DictConfig):
    from diffusha.algo.joint_trainer import JointCMTrainer
    set_random_seed(cfg.seed)
    joint_cm_model = hydra.utils.instantiate(cfg.joint_cm_model)
    if cfg.load_model:
        joint_cm_model = load_model(joint_cm_model,cfg.model_path,eval = False, device = cfg.device)
        print("Load CM Model from ",cfg.model_path)

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


@hydra.main(version_base = None,
                  config_path = '../../configs/maniskill/',
                  config_name = 'train_joint_combined')
def train_cm_v2(cfg:DictConfig):
    # Import required trainers
    from diffusha.algo.joint_trainer import JointEDMTrainer, JointCMTrainer
    from diffusha.algo.expert_trainer import ForwardTrainer
    from diffusha.model.diffusion_model import AdaptiveWeighting
    set_random_seed(cfg.seed)

    dataset = hydra.utils.instantiate(cfg.dataset)

    joint_edm_model = hydra.utils.instantiate(cfg.joint_edm_model)
    joint_edm_denosier = hydra.utils.instantiate(cfg.joint_edm_denoiser)
    weighting_model = AdaptiveWeighting(cfg.joint_edm_model.hidden_dim)
    joint_trainer = JointEDMTrainer(joint_edm_model, joint_edm_denosier, weighting_model=weighting_model, args = cfg)
    fwd_trainer = None
    joint_trainer.run_loop(cfg.n_iter, 
                         dataset, 
                         cfg.batch_size, 
                         cfg.gamma,
                         forward_trainer=fwd_trainer)
    joint_trainer.save()

    print("EDM Training Finished")

    set_random_seed(cfg.seed)
    joint_cm_model = hydra.utils.instantiate(cfg.joint_cm_model)
    target_model = deepcopy(joint_cm_model)
    joint_cm_denosier = hydra.utils.instantiate(cfg.joint_cm_denoiser)

    teacher_model = joint_edm_model
    teacher_denoiser = joint_edm_denosier
    weighting_model = None
    # import pdb; pdb.set_trace()
    joint_trainer = JointCMTrainer(joint_cm_model, 
                                joint_cm_denosier,
                                target_model, 
                                teacher_model,
                                teacher_denoiser, 
                                weighting_model=weighting_model,
                                args = cfg)
    fwd_cfg_path = os.path.join(hydra.utils.get_original_cwd(),cfg.fwd_trainer_cfg_path)            
    fwd_cfg = OmegaConf.load(fwd_cfg_path)
    fwd_cfg.forward_trainer.exp_name = cfg.exp_name
    fwd_cfg.forward_trainer.save_dir = cfg.save_dir
    fwd_model = hydra.utils.instantiate(fwd_cfg.forward_model)
    fwd_trainer = ForwardTrainer(fwd_model,fwd_cfg.forward_trainer)
    joint_trainer.run_loop(cfg.n_iter, 
                        dataset, 
                        cfg.batch_size,
                        cfg.gamma,
                        fwd_trainer)
    joint_trainer.save()
    fwd_trainer.save()
    print("CM Training Finished")
    


if __name__ == "__main__":
    # train_edm()
    # train_cm()
    # train_ddpm()
    # train_joint_edm()
    train_joint_cm()
    # train_cm_v2()


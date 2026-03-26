import warnings
from typing import Dict, Tuple, Union
import cv2
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union
import gymnasium as gym
import torch, random
import os
import h5py
import json
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
import copy 
import os 
import pickle
import torch 

from collections import defaultdict
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env(env_id, seed, idx, capture_video, run_name, **kwargs):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id,**kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id,**kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class GoalReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    goals: th.Tensor

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor




def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device

def get_obs_shape(
    observation_space: spaces.Space, n_env = 1
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        if n_env == 1:
            return observation_space.shape
        elif n_env > 1:
            return observation_space.shape[1:]
        
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space: spaces.Space,n_env = 1) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        if n_env == 1:
            return int(np.prod(action_space.shape))
        elif n_env > 1:
            return int(np.prod(action_space.shape[1:]))
        
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
    



def get_frame(env, episode, step, obs=None, reward=None, reward_sum=None, done = False, action: Optional[np.ndarray] = None,
              scale= None, frame= None, info=None, exp_name = None):
    # frame: (h, w, c)
    fontScale = .3

    env_name = env.unwrapped.spec.id

    if frame is None:
        rendered = env.render()
        if isinstance(rendered, th.Tensor):
            if rendered.device.type != "cpu":
                rendered = rendered.cpu()
            frame = np.ascontiguousarray(rendered.numpy(), dtype=np.uint8)
        else:
            frame = np.ascontiguousarray(rendered, dtype=np.uint8)
    if len(frame.shape) == 4:
        ## have a env num
        frame = frame[0,:,:,:]
    
    if 'Lunar' in env_name:
        if done:
            # print("info", info)
            # print("termination_info", info['termination_info'])
            if info is not None:
                if info['termination_info'] == "landed_at_site" or info['termination_info'] == "target_reached":
                    start_pt = (0, 0)
                    end_pt = (frame.shape[1]-1, frame.shape[0]-1)
                    cv2.rectangle(frame, start_pt, end_pt, (0, 255, 0), thickness=10)
                else:
                    start_pt = (0, 0)
                    end_pt = (frame.shape[1]-1, frame.shape[0]-1)
                    cv2.rectangle(frame, start_pt, end_pt, (255, 0, 0), thickness=10)
    
    if 'Peg' or 'Charge' in env_name:
        if done:
            if info is not None:
                if info['success']:
                    start_pt = (0, 0)
                    end_pt = (frame.shape[1]-1, frame.shape[0]-1)
                    cv2.rectangle(frame, start_pt, end_pt, (0, 255, 0), thickness=10)
                else:
                    start_pt = (0, 0)
                    end_pt = (frame.shape[1]-1, frame.shape[0]-1)
                    cv2.rectangle(frame, start_pt, end_pt, (255, 0, 0), thickness=10)
    
    # if reward is not None:
    #     frame = cv2.putText(frame, f'R: {reward:.4f}', org=(0, 60), fontFace=3, fontScale=fontScale, color=(0, 255, 0),
    #                         thickness=1)
    #     frame = cv2.putText(frame, f'R-sum: {reward_sum:.4f}', org=(0, 100), fontFace=3, fontScale=fontScale,
    #                         color=(0, 255, 0), thickness=1)
    # if exp_name is not None:
    #     frame = cv2.putText(frame, f'Exp Name: {exp_name}', org=(20, 20), fontFace=3, fontScale=fontScale, color=(0, 255, 0),
    #                         thickness=1)

    # if 'Push' not in env_name:
    #     if action is not None:
    #         frame = cv2.putText(frame, f'act 0: {action[0]:.3f}', org=(0, 140), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=2)
    #         frame = cv2.putText(frame, f'act 1: {action[1]:.3f}', org=(0, 180), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=2)

    #     if obs is not None:
    #         frame = cv2.putText(frame, f'obs[0]: {obs[0]:.3f}', org=(0, 140), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=2)
    #         frame = cv2.putText(frame, f'obs[1]: {obs[1]:.3f}', org=(0, 180), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=2)
    #         frame = cv2.putText(frame, f'obs[-2], obs[-1]: {obs[-2]:.3f}, {obs[-1]:.3f}', org=(0, 220), fontFace=3,
    #                             fontScale=fontScale, color=(0, 255, 0), thickness=2)

    #     if info is not None:
    #         frame = cv2.putText(frame, f'game-over: {info.get("game_over_reason", "")}', org=(0, 260), fontFace=3,
    #                             fontScale=fontScale, color=(0, 255, 0), thickness=2)
    #         frame = cv2.putText(frame, f'timelimit: {info.get("TimeLimit.truncated", "")}', org=(0, 300), fontFace=3,
    #                             fontScale=fontScale, color=(0, 255, 0), thickness=2)

    # else:
    #     # import pdb; pdb.set_trace()
    #     if info is not None:
    #         frame = cv2.putText(frame, f"state: {info['state']}", org=(0, 140), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=1)

    #     with np.printoptions(precision=2, suppress=True):
    #         frame = cv2.putText(frame, f"block: {obs[:3]}", org=(0, 160), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=1)
    #         frame = cv2.putText(frame, f"ee: {obs[3:5]}", org=(0, 180), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=1)
    #         frame = cv2.putText(frame, f"action: {action}", org=(0, 200), fontFace=3, fontScale=fontScale,
    #                             color=(0, 255, 0), thickness=1)

    # if obs is not None:
    #     x, y, vx, vy, rx, ry, rvx, rvy, *_ = obs
    #     frame = cv2.putText(frame, f'(X, Y): ({x:.1f}, {y:.1f})  (VX, VY): ({vx:.1f}, {vy:.1f})', org=(0, 140), fontFace=3, fontScale=fontScale, color=(0, 255, 0), thickness=2)
    #     frame = cv2.putText(frame, f'(RX, RY): ({rx:.1f}, {ry:.1f})  (RVX, RVY): ({rvx:.1f}, {rvy:.1f})', org=(0, 180), fontFace=3, fontScale=fontScale, color=(0, 255, 0), thickness=2)

    # TMP: Luzhe
    output_dir = f"paper_videos/{env_name}/{exp_name}/EP_{episode}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/STEP_{step}.png"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, frame_rgb)

    frame = cv2.putText(frame, f'EP: {episode} STEP: {step}', org=(10, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                        color=(0, 255, 0), thickness=2)


    if scale is not None:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, dsize=(width, height))

    return frame


def parse_motionplanning_data(h5_file_path,save_dir,save_name,env_id = 'peg'):
    # state 
    data_stats = defaultdict(list)
    obs_data = defaultdict(list)
    obs2_data = defaultdict(list)
    goal_data = defaultdict(list)
    action_data = []
    if env_id == 'peg':
        obs_names = ["qpos","qvel","tcp_pose","peg_pose","peg_half_size"]
        goal_names = ["box_hole_pose","box_hole_radius"]
    elif env_id == 'charger':
        obs_names = ["qpos","qvel","tcp_pose","charger_pose"]
        goal_names = ["receptacle_pose","goal_pose"]


    def explore_group(group, level=0):
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                data = item[:]
                if key in obs_names:
                    obs_data[key].append(data[0:-1])
                    obs2_data[key].append(data[1:])
                elif key in goal_names:
                    goal_data[key].append(data[0:-1])
                elif key == 'actions':
                    action_data.append(data)
                    data_stats['length'].append(data.shape[0])
                    action_norm = np.linalg.norm(data,axis = -1)
                    data_stats['action_norm_min'].append(np.min(action_norm))
                    data_stats['action_norm_max'].append(np.max(action_norm))
                    data_stats['action_norm_mean'].append(np.mean(action_norm))

            elif isinstance(item, h5py.Group):
                # If it's a group, recurse into it
                explore_group(item, level+1)

    h5_file = h5py.File(h5_file_path)
    keys = list(h5_file.keys()) ## index of trajectories 
    for k in keys:
        f = h5_file[k]
        explore_group(f,level=0)

    ## concat data
    for k,v in obs_data.items():
        obs_data[k] = np.concatenate(v,axis = 0)
    for k,v in obs2_data.items():
        obs2_data[k] = np.concatenate(v,axis = 0)
    for k,v in goal_data.items():
        goal_data[k] = np.concatenate(v,axis=0)
    action_data = np.concatenate(action_data,axis=0)

    ## prepare training data 
    obs = []
    obs2 = []
    goal = []
    for k in obs_names:
        obs.append(obs_data[k])
        obs2.append(obs2_data[k])
    obs = np.concatenate(obs,axis=-1)
    obs2 = np.concatenate(obs2,axis=-1)
    for k in goal_names:
        v = goal_data[k]
        if len(v.shape) == 1:
            v = v.reshape(-1,1)
        goal.append(v)
    goal = np.concatenate(goal,axis=-1)
    file_name = save_dir +  f"/{save_name}.pkl"
    os.makedirs(save_dir, exist_ok=True)

    buffer = np.concatenate([obs,action_data,obs2,goal],axis = -1)
    # torch.save(buffer,file_name)
    return obs,action_data,obs2,goal,data_stats




class UR5_DataPreprocess:
    # class-level attribute
    state_keys = ['gripper_pos', 'gripper_quat']
    action_keys = ['cmd_trans_vel', 'cmd_rot_vel']
    data_dims = {
        'step':1,
        'joint_states': 6,
        'gripper_pos': 3,
        'gripper_quat': 4,
        'cmd_trans_vel': 3,
        'cmd_rot_vel': 3,
        'cmd_grasp_pos': 1
    }
    dt = 1.0/30.0
    def __init__(self):
        pass

    @classmethod 
    def __filter_position(cls,positions, sample_frequency, cutoff_frequency = 2, filter_order = 4):
        nyquist_freq = 0.5 * sample_frequency
        normalized_cutoff = cutoff_frequency / nyquist_freq
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        filtered_positions = np.zeros_like(positions)
        for dim in range(3):
            filtered_positions[:, dim] = filtfilt(b, a, positions[:, dim], axis=0)
        return filtered_positions
    
    @classmethod 
    def __flip_rvec(cls,rvec):
        flipped_rvec = np.zeros_like(rvec)
        flipped_rvec[0] = rvec[0]
        for i in range(1,len(rvec)):
            if np.dot(rvec[i], flipped_rvec[i-1]) < 0:
                flipped_rvec[i] = -rvec[i]
            else:
                flipped_rvec[i] = rvec[i]
        return flipped_rvec
    
    @classmethod 
    def __filter_orientation(cls,quat, sample_frequency, cutoff_frequency = 1, filter_order = 4 ): 
        rot = Rotation.from_quat(quat)
        rvec = rot.as_rotvec()
        rvec = cls.__flip_rvec(rvec)
        nyquist = 0.5 * sample_frequency
        normalized_cut = cutoff_frequency / nyquist
        b, a = butter(filter_order, normalized_cut, btype='low', analog=False)
        
        rvecs_f = np.zeros_like(rvec)
        for dim in range(3):
            rvecs_f[:, dim] = filtfilt(b, a, rvec[:, dim], axis=0)
        rot = Rotation.from_rotvec(rvecs_f)
        quat_f = rot.as_quat()
        return quat_f
        


    @classmethod
    def _list2dict(cls,list_dict):
        ## prepare the dictionary
        tmp_dict = dict()
        tmp_dict['step'] = [] ## for missing steps 
        for key in cls.state_keys:
            tmp_dict[key] = []
        for key in cls.action_keys:
            tmp_dict[key] = []
        ## traverse the list of dictionary
        for i,d in enumerate(list_dict):
            red_flag = False ## don't know why there could be some missing data 
            for key in tmp_dict.keys():
                v = d[key]
                if type(v) == int or type(v) == float:
                    d[key] = np.array([v]).reshape(1,)
                else:
                    d[key] = np.array(v)

                if d[key].shape[0] != cls.data_dims[key]:
                    #! subscribe to wrong topic 
                    red_flag = True
            if not red_flag:
                for key in tmp_dict.keys():
                    tmp_dict[key].append(d[key]) 
            
                
        for k,v in tmp_dict.items():
            if len(v) == 0:
                return False, None
            
        for key in tmp_dict.keys():
            tmp_dict[key] = np.stack(tmp_dict[key], axis=0)
        
        ## Check the dimension and shape
        n_sample = tmp_dict['step'].shape[0]
        for key in tmp_dict.keys():
            assert tmp_dict[key].shape[1] == cls.data_dims[key]
            assert tmp_dict[key].shape[0] == n_sample
        
        return True, tmp_dict

    """Important functions for data extraction
        1. define the dimension of the ur5 data (state_dim, action_dim)
    """
    @classmethod
    def _filter_action(cls,raw_d:dict,trans_threshold = 0.001, rot_threshold = 0.001, max_rot_vel = 0.3):
        ## 1. Filter action where |action_norm| is close to zero 
        ## unintentional stop during data collection
        import copy 
        d = copy.deepcopy(raw_d)
        cmd_trans_vel = d['cmd_trans_vel']
        trans_vel_norm = np.linalg.norm(cmd_trans_vel,axis=-1)
        rot_vel_norm = np.linalg.norm(d['cmd_rot_vel'],axis=-1)
        kept_index = np.where((trans_vel_norm + rot_vel_norm)> trans_threshold)[0]
        for k,v in d.items():
            d[k] = v[kept_index]
        
        ## concatenate the data to fill out the empty

        n_valid_steps = len(kept_index)
        d['step'] = np.arange(n_valid_steps).reshape(-1,1)


        ## 2. Smooth action rot vel
        ## make sure in the training data, there is no unwanted rotation velocity
        ## we respect the original translation and position, as they serve as the diversiy
        if 'cmd_rot_vel' in d.keys():
            d['cmd_rot_vel'] = np.clip(d['cmd_rot_vel'], -max_rot_vel, max_rot_vel)
        ## 3. Output the data dictionary, which could be used for training, while it's necessary to add noise on the state
        return d 

    
    @classmethod 
    def __interpolate_position(cls,position, timesteps):
        t_samples = timesteps
        x_spline = CubicSpline(t_samples, position[:, 0], bc_type='natural')
        y_spline = CubicSpline(t_samples, position[:, 1], bc_type='natural')
        z_spline = CubicSpline(t_samples, position[:, 2], bc_type='natural')
        def p(t):
            return np.array([
                x_spline(t),
                y_spline(t),
                z_spline(t)
            ])
        # spline.derivative() returns the first derivative as another spline
        x_dot = x_spline.derivative()
        y_dot = y_spline.derivative()
        z_dot = z_spline.derivative()
        def v(t):
            return np.array([
                x_dot(t),
                y_dot(t),
                z_dot(t)
            ])
        return p, v
    
    @classmethod 
    def __resample_pos_vel(cls,position,timesteps):
        p_func,v_func = cls.__interpolate_position(position,timesteps)
        p_list,v_list = [],[]
        for t in timesteps:
            p_list.append(p_func(t))
            v_list.append(v_func(t))
        return np.stack(p_list,axis=0),np.stack(v_list,axis=0)
    
    @classmethod
    def __approxiamte_angular_velocity_matrix_center_diff(cls,rvec,t_array):
        rot = Rotation.from_rotvec(rvec)    
        N = len(t_array)
        w_array = np.zeros((N, 3))
        R = rot.as_matrix()
        for i in range(1,N-1):
            R_prev = R[i-1]
            R_next = R[i+1]
            R_diff = R_next @ R_prev.T
            angle = np.arccos((np.trace(R_diff) - 1) / 2)
            axis = np.array([
                R_diff[2, 1] - R_diff[1, 2],
                R_diff[0, 2] - R_diff[2, 0],
                R_diff[1, 0] - R_diff[0, 1]
            ]) / (2 * np.sin(angle) if angle != 0 else 1)
            angular_velocity = (axis * angle) / (t_array[i+1] - t_array[i-1])
            w_array[i] = angular_velocity
        w_array[0] = w_array[1]
        w_array[-1] = w_array[-2]
        return w_array
    
    @classmethod 
    def __resample_orentation_vel(cls,quat, timesteps):
        rot = Rotation.from_quat(quat)
        slerp = Slerp(timesteps, rot)
        rot_list = slerp(timesteps)
        rvec_list = rot_list.as_rotvec() 
        w_list = cls.__approxiamte_angular_velocity_matrix_center_diff(rvec_list,timesteps)
        return rot_list.as_quat(), w_list

    
    @classmethod
    def preprocess(cls,raw_d:dict,if_smooth = True, if_filter_action = True):
        d = copy.deepcopy(raw_d)
        ## 1. shift the steps to zero 
        steps = d['step']
        min_step = steps[0]
        d['step'] = d['step'] - min_step 
        ## 2. filter the action
        if if_filter_action:
            d = cls._filter_action(d)
            
        d['timesteps'] = d['step'].reshape(-1) * cls.dt 
        if if_smooth:
            d['gripper_pos'] = cls.__filter_position(d['gripper_pos'], sample_frequency=1.0/cls.dt)
            d['gripper_quat'] = cls.__filter_orientation(d['gripper_quat'], sample_frequency=1.0/cls.dt)
            ## 4. interpolate the smoothed data to get smooth trans_vel and rot_vel
            new_p,new_v = cls.__resample_pos_vel(d['gripper_pos'],d['timesteps'])
            new_quat, new_w = cls.__resample_orentation_vel(d['gripper_quat'],d['timesteps'])
            ## 5. clip the rot_vel
            new_w = np.clip(new_w, -0.3, 0.3)
            ## 6. output the data dictionary (we won't change the sample frequency, so the maximum step remain the same, the total duration time varies)
            d['gripper_pos'] = new_p
            d['gripper_quat'] = new_quat
            d['cmd_trans_vel'] = new_v
            d['cmd_rot_vel'] = new_w
        return d 

    @classmethod
    def _prolong_interpolate_state(cls,raw_d:dict, prolong_ratio):
        d = copy.deepcopy(raw_d)
        ## 1. shift the steps to zero 
        steps = d['step']
        min_step = steps[0]
        d['step'] = d['step'] - min_step 
        d['timesteps'] = d['step'].reshape(-1) * cls.dt 
        ## 2. times the step with prolong_ratio 
        d['timesteps'] = d['timesteps'] * prolong_ratio 
        ## 3. Now we need to smooth the trajectory so that we can use the frame difference 
        d['gripper_pos'] = cls.__filter_position(d['gripper_pos'], sample_frequency=1.0/cls.dt)
        d['gripper_quat'] = cls.__filter_orientation(d['gripper_quat'], sample_frequency=1.0/cls.dt)
        ## 4. interpolate the smoothed data to get smooth trans_vel and rot_vel
        new_p,new_v = cls.__resample_pos_vel(d['gripper_pos'],d['timesteps'])
        new_quat, new_w = cls.__resample_orentation_vel(d['gripper_quat'],d['timesteps'])

        ## 6. output the data dictionary (we won't change the sample frequency, so the maximum step remain the same, the total duration time varies)
        d['gripper_pos'] = new_p
        d['gripper_quat'] = new_quat
        d['cmd_trans_vel'] = new_v
        d['cmd_rot_vel'] = new_w
        return d 

    @classmethod
    def _add_oreintation_perturbation(cls,raw_d:dict, time_index, max_deg, tau):
        ## 1. input the smoothed data, we only care about state, since we gonna change the velocity. smoothed because we need to calculate the velocity 
        ## 2. add delta orientation at step t
        d = copy.deepcopy(raw_d)
        quat = d['gripper_quat']
        disturbed_quat, start_idx, end_idx  = cls.disturb_quaternion_trajectory(quat, d['timesteps'], time_index, max_deg, tau)
        disturbed_quat = disturbed_quat[start_idx:end_idx]
        ## 3. generate the recovery action starting from t to t' where orientation comes back 
        for k,v in d.items():
            d[k] = v[start_idx:end_idx]
        new_quat,new_w = cls.__resample_orentation_vel(disturbed_quat,d['timesteps'])
        d['gripper_quat'] = new_quat
        d['cmd_rot_vel'] = new_w
        return d 

    @classmethod
    def disturb_quaternion_trajectory(
            cls,
            quat: np.ndarray,
            timesteps: np.ndarray,
            t_index: int,
            max_deg: float,
            tau: float
        ) -> np.ndarray:
    
        assert quat.shape[0] == timesteps.shape[0], "shape mismatch"
        N = quat.shape[0]
        disturbed_quat = quat.copy()
        idx = t_index

        ## 3. generate perturbation
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        max_rad = np.deg2rad(max_deg)
        theta = np.random.uniform(low=0.0, high=max_rad)
        r_perturb = Rotation.from_rotvec(axis * theta)

        ## 4. Add Perturbation
        r_orig = Rotation.from_quat(disturbed_quat[idx])  
        r_disturbed = r_perturb * r_orig
        disturbed_quat[idx] = r_disturbed.as_quat()

        def slerp_step_scipy(q0, q1, alpha):
            rots = Rotation.from_quat(np.array([q0, q1]))
            slerp = Slerp([0, 1], rots)
            rot_out = slerp([alpha])  
            return rot_out.as_quat()[0]  
        
        start_idx = idx 
        end_idx = N - 1
        for i in range(idx, N - 1):
            dt_i = timesteps[i+1] - timesteps[i]
            if dt_i < 0:
                dt_i = 0  
            alpha = 1.0 - np.exp(- dt_i / tau)
            q_i = disturbed_quat[i]
            q_ref_next = quat[i+1]  
            q_next = slerp_step_scipy(q_i, q_ref_next, alpha)
            disturbed_quat[i+1] = q_next
            # if disturbed_quat is close to the reference, then break
            if np.linalg.norm(q_next - q_ref_next) < 0.001 and (i > start_idx + 10):
                end_idx = i
                break
        return disturbed_quat, start_idx, end_idx
    
    @classmethod 
    def _add_translation_perturbation(cls,raw_d:dict, time_index, max_disturbance, tau):
        ## 1. input smoothed data
        d = copy.deepcopy(raw_d)
        pos = d['gripper_pos']
        ## 2. add delta translation at step t 
        disturbed_pos,start_idx,end_idx = cls.disturb_position_trajectory(pos, d['timesteps'], time_index, max_disturbance, tau)
        disturbed_pos = disturbed_pos[start_idx:end_idx]
        ## 3. generate the recovery action starting from t to t' where orientation comes back 
        for k,v in d.items():
            d[k] = v[start_idx:end_idx]
        new_pos,new_vel = cls.__resample_pos_vel(disturbed_pos,d['timesteps'])
        d['gripper_pos'] = new_pos
        d['cmd_trans_vel'] = new_vel
        return d
        
    def disturb_position_trajectory(
        pos: np.ndarray,
        timesteps: np.ndarray,
        t: int,
        max_disturbance: float,
        tau: float
    ) -> np.ndarray:
        
        assert pos.shape[0] == timesteps.shape[0], "shape mismatch"
        N = pos.shape[0]
        disturbed_pos = pos.copy()
        idx = t
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction) 
        
        disturbance_mag = np.random.uniform(low=0.0, high=max_disturbance)
        disturbance_vec = direction * disturbance_mag
        disturbed_pos[idx] += disturbance_vec
        start_idx = idx 
        end_idx = N - 1
        for i in range(idx, N - 1):
            dt_i = timesteps[i+1] - timesteps[i]
            if dt_i < 0:
                dt_i = 0.0 

            alpha = 1 - np.exp(- dt_i / tau)  
            disturbed_pos[i+1] = (1 - alpha) * disturbed_pos[i] + alpha * pos[i+1]
            if np.linalg.norm(disturbed_pos[i+1] - pos[i+1]) < 0.001 and (i > start_idx + 10):
                end_idx = i
                break
        return disturbed_pos,start_idx,end_idx

    @classmethod
    def _ur5_rawdata2transition(cls,data_dict, remove_start = 0, remove_end = 1):
        ## state = (joint_states, gripper_pos, gripper_quat)
        ## action = (cmd_trans_vel, cmd_rot_vel), if act_include_grasp_pos = True, then add cmd_grasp_pos 

        state = np.concatenate([data_dict[k] for k in cls.state_keys], axis=-1)
        action = np.concatenate([data_dict[k] for k in cls.action_keys], axis=-1)
        ## Prepare transition
        cur_state = state[:-1]
        cur_action = action[:-1]
        next_state = state[1:]   
        transition_dict = {'obs': cur_state, 'act': cur_action, 'obs2': next_state}
        for key in transition_dict.keys():
            transition_dict[key] = transition_dict[key][remove_start:-remove_end]

        return transition_dict
    
    @classmethod
    def _dump_dict(cls,src_dict, dst_dict):
        for key in src_dict.keys():
            dst_dict[key].append(src_dict[key]) ### list of np.array with different size
        return dst_dict
    
    @classmethod
    def ur5_traj_extract(cls,list_file_path):
        # input a list of file_dir where there are many pkl files
        list_of_dict = []
        for path in list_file_path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".pt"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            print(f"Loading {file_path}")
                            tmp_list = pickle.load(f) ## list of dictionary
                            if len(tmp_list) == 0:    ## empty file
                                continue
                            success,tmp_dict = cls._list2dict(tmp_list)
                            if success:
                                list_of_dict.append(tmp_dict)
        return list_of_dict

    @classmethod    
    def traj_aug_prolong(cls,raw_d:dict,n_prolong = 5, ratio = 1.5):
        assert 0.0 < ratio < 2.0, "ratio should be in (0,2)"
        # 1. the prolonged velocity is centered at original velocity and the max_ratio is the max ratio of the velocity
        d = copy.deepcopy(raw_d)
        if ratio > 1.0:
            prolong_ratios = np.random.uniform(low = 1.0, high=ratio, size=n_prolong)
        else:
            prolong_ratios = np.random.uniform(low = ratio, high=1, size=n_prolong)
        # 2. clip the ratio with 
        list_dicts = []
        for ratio in prolong_ratios:
            d = cls._prolong_interpolate_state(d, ratio)
            list_dicts.append(d)
        return list_dicts
    @classmethod
    def traj_aug_perturb(cls,raw_d:dict,n_perturbs, with_trans = True, with_rot = True, max_disturbance = 0.05, max_deg = 10, tau = 0.1):
        d = copy.deepcopy(raw_d)
        n_steps = raw_d['step'].shape[0]
        # step_indexs = np.random.choice(n_steps, size=n_perturbs, replace=False)
        step_indexs = np.random.randint(0, n_steps - 40, size=n_perturbs) ## not last 2 seconds
        list_dicts = []
        for idx in step_indexs:
            if with_trans:
                list_dicts.append(cls._add_translation_perturbation(d, idx, max_disturbance, tau))
            if with_rot:
                list_dicts.append(cls._add_oreintation_perturbation(d, idx, max_deg, tau))
        return list_dicts

    @classmethod
    def ur5_trajs2data(cls,list_dicts,save_dir,new_name, if_save = True):
        data = {
            'obs': [],
            'act': [],
            'obs2': []
        }
        for d in list_dicts:
            transition_dict = cls._ur5_rawdata2transition(d)
            data = cls._dump_dict(transition_dict, data)
        for k,v in data.items():
            data[k] = np.concatenate(v, axis=0)
            print("Save Data: ", k, data[k].shape)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f"{new_name}.pkl")
        # Concatenate the buffer data and dump it
        _current_buffer = np.concatenate([data['obs'], data['act'],data['obs2']], axis=-1)
        if if_save:
            print(f"Dumping buffer to {file_name}")
            torch.save(_current_buffer, file_name)
        else:
            return _current_buffer
    @classmethod
    def ur5_data_extract(cls,list_file_path,save_dir, new_name = 'ur5_data', remove_start = 0, remove_end = 1):
        """
        Input: [path1, path2, ...]
        in each path, there are multiple .pt files, use pickle to load them and concatenate them
        """
        
        data = {
            'obs': [],
            'act': [],
            'obs2': []
        }
        for path in list_file_path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".pt"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            tmp_list = pickle.load(f) ## list of dictionary
                            if len(tmp_list) == 0:    ## empty file
                                continue
                            success,tmp_dict = cls._list2dict(tmp_list) 
                            if success:
                                transition_dict = cls._ur5_rawdata2transition(tmp_dict, act_include_grasp_pos = False, remove_start = remove_start, remove_end = remove_end)
                                data = cls._dump_dict(transition_dict, data)
                            else:
                                print(f"Failed to load data from {file_path}")

        for k,v in data.items():
            data[k] = np.concatenate(v, axis=0)
            print("Load Data: ", k, data[k].shape)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f"{new_name}.pkl")
        # Concatenate the buffer data and dump it
        _current_buffer = np.concatenate([data['obs'], data['act'],data['obs2']], axis=-1)
        print(f"Dumping buffer to {file_name}")
        torch.save(_current_buffer, file_name)
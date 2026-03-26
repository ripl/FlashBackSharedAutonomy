from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


# @register_env("SimpleChargerPlug-v1", max_episode_steps=800)
@register_env("SimpleChargerPlug-v1", max_episode_steps=300)
class SimpleChargerPlug(BaseEnv):
    """
    **Task Description:**
    The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

    **Randomizations:**
    - The charger position is randomized on the XY plane on top of the table. The rotation is also randomized
    - The receptacle position is randomized on the XY plane and the rotation is also randomized. Note that the human render camera has its pose
    fixed relative to the receptacle.

    **Success Conditions:**
    - The charger is inserted into the receptacle
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlugCharger-v1_rt.mp4"

    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    # _clearance = 5e-3  # single side clearance
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if 'fix_size' in kwargs:
            self.fix_size = kwargs['fix_size']
            kwargs.pop('fix_size')
        else:
            self.fix_size = False 
        if 'random_initial_position' in kwargs:
            self.random_initial_position = kwargs['random_initial_position']
            kwargs.pop('random_initial_position')
        else:
            self.random_initial_position = False 
        
        if 'simple_target_position' in kwargs:
            self.simple_target_position = kwargs['simple_target_position']
            kwargs.pop('simple_target_position')
        else:
            self.simple_target_position = False 

        self.prev_act_dim = 8 if kwargs['control_mode'] == 'pd_joint_delta_pos' else 7
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
         ### goal dim for expert and diffusion policy
        self.goal_dim = 14
        self.action_dim = self.single_action_space.shape[-1]

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize agent
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                torch.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            if self.random_initial_position:
                xy = randomization.uniform(
                    [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self._base_size[2]
                ori = randomization.random_quaternions(
                    n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
                )
            else:
                xy = randomization.uniform(
                    [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self._base_size[2]
                ori = randomization.random_quaternions(
                    n=b, lock_x=True, lock_y=True, bounds=(0, 0)
                )

            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            if self.simple_target_position:
                # xy = randomization.uniform([0.01, 0.0], [0.1, 0.0], size=(b, 2))
                xy_x_options = torch.tensor([0.01, 0.1])
                xy_x = xy_x_options[torch.randint(0, len(xy_x_options), (b,))]

                xy_y = torch.full((b,), 0.0)

                # Build xy
                xy = torch.stack((xy_x, xy_y), dim=1)

                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = 0.1
                ori = randomization.random_quaternions(
                    n=b,
                    lock_x=True,
                    lock_y=True,
                    bounds=(torch.pi, torch.pi),
                )
            else:
                xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = 0.1
                ori = randomization.random_quaternions(
                    n=b,
                    lock_x=True,
                    lock_y=True,
                    bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
                )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        # success = (obj_to_goal_dist <= 0.018) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                receptacle_pose=self.receptacle.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
            )
        return obs
    
    def _smooth_action_penalty(self,action):
        ## First Order
        action_norm = torch.linalg.norm(action, axis = -1) ## in averge 1.4 

        ## Second Order 
        delta_action_norm = torch.linalg.norm( (action - self.previous_actions), axis = -1 ) 

        return action_norm * 0.05 + delta_action_norm * 0.01

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        ## Stage 1 
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.charger.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose * (offset)
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)
        
        ## Stage 2
        is_grasped = self.agent.is_grasping(self.charger, max_angle=20)
        reward = reaching_reward + is_grasped

        ## Stage 3 
        charger_wrt_goal = self.goal_pose.inv() * self.charger.pose 
        charger_wrt_goal_dist = torch.linalg.norm(charger_wrt_goal.p, axis = 1)
        pre_insertion_reward = 3 * ( 1 - torch.tanh(5 * charger_wrt_goal_dist))
        reward += is_grasped * pre_insertion_reward

        ## Add action penalty
        action_penalty = (action**2).sum(-1) * 0.02 ## <---- parameter
        reward -= action_penalty

        ## Stage 4 
        eval_res = self.evaluate()
        reward[eval_res['success']] = 10
        return reward 


    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # max_reward = 1.0
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    def _parse_obs(self,obs):
        agent_info = obs['agent']
        qpos,qvel = agent_info['qpos'],agent_info['qvel'] ### 7 + 7
        extra_info = obs['extra'] 
        tcp_pose = extra_info['tcp_pose'] ## 7 
        ### peg info 
        charger_pose = extra_info['charger_pose'] ## 7
        receptacle_pose = extra_info['receptacle_pose']
        goal_pose = extra_info['goal_pose']

        goal = torch.cat([receptacle_pose,goal_pose],dim=-1)
        state = torch.cat(
            [qpos,qvel,tcp_pose,charger_pose], dim=-1
        )
        return state, goal 

    def step(self,act):
        obs,reward, terminated, trunctated, info = super().step(act)
        self.previous_actions = act
        info['raw_reward'] = reward.clone()
        info['penalty'] = 0.0
        state,goal = self._parse_obs(obs)
        info['goal'] = goal
        return state, reward, terminated,trunctated,info
    
    def _reset_previous_actions(self,options):
        if options is None:
            self.previous_actions = torch.zeros((self.num_envs,self.prev_act_dim)).to(self.device)
        elif 'env_idx' in options:
            self.previous_actions[options['env_idx'],:] = 0
    
    def reset(self,seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        obs, info = super().reset(seed, options)
        ## record previous action 
        self._reset_previous_actions(options)
        state, goal = self._parse_obs(obs)
        info['goal'] = goal 
        return state, info


@register_env("CurriculumChargerPlug-v1", max_episode_steps=300)
class CurriculumChargerPlug(BaseEnv):
    """
    **Task Description:**
    The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

    **Randomizations:**
    - The charger position is randomized on the XY plane on top of the table. The rotation is also randomized
    - The receptacle position is randomized on the XY plane and the rotation is also randomized. Note that the human render camera has its pose
    fixed relative to the receptacle.

    **Success Conditions:**
    - The charger is inserted into the receptacle
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlugCharger-v1_rt.mp4"

    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    # _clearance = 5e-3  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Curriculum: init clearance、target clearance、processed clearance
        # self.initial_eval_threshold = 5e-2  # loose the success check at init
        self.initial_eval_threshold = 0.018  # loose the success check at init
        self.target_eval_threshold = 5e-3  # gradually tighten the success check
        self.curriculum_progress = 0.0  # process:range 0.0 ~ 1.0
        self.curriculum_eval_threshold = self.initial_eval_threshold  # current clearance

        if 'fix_size' in kwargs:
            self.fix_size = kwargs['fix_size']
            kwargs.pop('fix_size')
        else:
            self.fix_size = False 
        if 'random_initial_position' in kwargs:
            self.random_initial_position = kwargs['random_initial_position']
            kwargs.pop('random_initial_position')
        else:
            self.random_initial_position = False 
        
        if 'simple_target_position' in kwargs:
            self.simple_target_position = kwargs['simple_target_position']
            kwargs.pop('simple_target_position')
        else:
            self.simple_target_position = False 

        self.prev_act_dim = 8 if kwargs['control_mode'] == 'pd_joint_delta_pos' else 7
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
         ### goal dim for expert and diffusion policy
        self.goal_dim = 14
        self.action_dim = self.single_action_space.shape[-1]
    
    def set_initial_eval_threshold(self, threshold):
        self.initial_eval_threshold = threshold

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        # print("physical receptacle clearance: ", self.curriculum_clearance)
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize agent
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                torch.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            if self.random_initial_position:
                xy = randomization.uniform(
                    [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self._base_size[2]
                ori = randomization.random_quaternions(
                    n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
                )
            else:
                xy = randomization.uniform(
                    [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self._base_size[2]
                ori = randomization.random_quaternions(
                    n=b, lock_x=True, lock_y=True, bounds=(0, 0)
                )

            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            if self.simple_target_position:
                # xy = randomization.uniform([0.01, 0.0], [0.1, 0.0], size=(b, 2))
                xy_x_options = torch.tensor([0.01, 0.1])
                xy_x = xy_x_options[torch.randint(0, len(xy_x_options), (b,))]

                xy_y = torch.full((b,), 0.0)

                # Build xy
                xy = torch.stack((xy_x, xy_y), dim=1)

                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = 0.1
                ori = randomization.random_quaternions(
                    n=b,
                    lock_x=True,
                    lock_y=True,
                    bounds=(torch.pi, torch.pi),
                )
            else:
                xy = randomization.uniform([0.01, -0.5], [0.1, 0.1], size=(b, 2))
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = 0.1
                ori = randomization.random_quaternions(
                    n=b,
                    lock_x=True,
                    lock_y=True,
                    bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
                )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
            """
            Evaluate the success of the task based on the distance and angle to the goal.
            """
            obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
            success = (obj_to_goal_dist <= self.curriculum_eval_threshold) & (obj_to_goal_angle <= 0.2)
            return dict(
                obj_to_goal_dist=obj_to_goal_dist,
                obj_to_goal_angle=obj_to_goal_angle,
                success=success,
            )

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                receptacle_pose=self.receptacle.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
            )
        return obs
    
    def _smooth_action_penalty(self,action):
        ## First Order
        action_norm = torch.linalg.norm(action, axis = -1) ## in averge 1.4 

        ## Second Order 
        delta_action_norm = torch.linalg.norm( (action - self.previous_actions), axis = -1 ) 

        return action_norm * 0.05 + delta_action_norm * 0.01

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        ## Stage 1 
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.charger.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose * (offset)
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)
        
        ## Stage 2
        is_grasped = self.agent.is_grasping(self.charger, max_angle=20)
        reward = reaching_reward + is_grasped

        ## Stage 3 
        charger_wrt_goal = self.goal_pose.inv() * self.charger.pose 
        charger_wrt_goal_dist = torch.linalg.norm(charger_wrt_goal.p, axis = 1)
        pre_insertion_reward = 3 * ( 1 - torch.tanh(5 * charger_wrt_goal_dist))
        reward += is_grasped * pre_insertion_reward

        ## Add action penalty
        action_penalty = (action**2).sum(-1) * 0.02 ## <---- parameter
        reward -= action_penalty

        ## Stage 4 
        eval_res = self.evaluate()
        reward[eval_res['success']] = 10
        return reward 


    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # max_reward = 1.0
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    def _parse_obs(self,obs):
        agent_info = obs['agent']
        qpos,qvel = agent_info['qpos'],agent_info['qvel'] ### 7 + 7
        extra_info = obs['extra'] 
        tcp_pose = extra_info['tcp_pose'] ## 7 
        ### peg info 
        charger_pose = extra_info['charger_pose'] ## 7
        receptacle_pose = extra_info['receptacle_pose']
        goal_pose = extra_info['goal_pose']


        goal = torch.cat([receptacle_pose,goal_pose],dim=-1)
        state = torch.cat(
            [qpos,qvel,tcp_pose,charger_pose], dim=-1
        )
        return state, goal 

    def step(self,act):
        obs,reward, terminated, trunctated, info = super().step(act)
        self.previous_actions = act
        info['raw_reward'] = reward.clone()
        info['penalty'] = 0.0
        state,goal = self._parse_obs(obs)
        info['goal'] = goal
        return state, reward, terminated,trunctated,info
    
    def _reset_previous_actions(self,options):
        if options is None:
            self.previous_actions = torch.zeros((self.num_envs,self.prev_act_dim)).to(self.device)
        elif 'env_idx' in options:
            self.previous_actions[options['env_idx'],:] = 0
    
    def reset(self,seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        obs, info = super().reset(seed, options)
        ## record previous action 
        self._reset_previous_actions(options)
        state, goal = self._parse_obs(obs)
        info['goal'] = goal 
        return state, info
    
    def update_curriculum(self, progress: float):
        """
        Update the clearance based on the progress of the curriculum
        """
        self.curriculum_progress = progress
        self.curriculum_eval_threshold = (
            self.initial_eval_threshold
            + (self.target_eval_threshold - self.initial_eval_threshold) * progress
        )
        print(f"Updated eval threshold: {self.curriculum_eval_threshold}")
    
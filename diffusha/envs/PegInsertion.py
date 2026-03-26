from ast import If
from typing import Any, Dict, Union

from matplotlib import axis
import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig


def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


# @register_env("SimplePegInsertion-v1", max_episode_steps=300)
@register_env("SimplePegInsertion-v1", max_episode_steps=200)
class SimplePegInsertion(BaseEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and insert the orange end into the box with a hole in it.

    **Randomizations:**
    - Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value. (during reconfiguration)
    - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.003m of clearance. (during reconfiguration)
    - Peg is laid flat on table and has it's xy position and z-axis rotation randomized
    - Box is laid flat on table and has it's xy position and z-axis rotation randomized

    **Success Conditions:**
    - The white end of the peg is within 0.015m of the center of the box (inserted mid way).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    # _clearance = 0.003
    _clearance = 0.01

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        ## Only supprt ee_delta control; state_dict obs 
        # assert 'pd_ee_delta_pose' == kwargs['control_mode'] and 'state_dict' == kwargs['obs_mode'], kwargs
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
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
        self.penalty_clip_ratio = 0.1
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )
        ### goal dim for expert and diffusion policy
        self.goal_dim = 8 

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            if not self.fix_size:
                lengths = self._batched_episode_rng.uniform(0.085, 0.125)
                radii = self._batched_episode_rng.uniform(0.015, 0.025)
                centers = (
                    0.5
                    * (lengths - radii)[:, None]
                    * self._batched_episode_rng.uniform(-1, 1, size=(2,))
                )
            else:
                lengths = self._batched_episode_rng.uniform(0.1, 0.1)
                radii = self._batched_episode_rng.uniform(0.02, 0.02)
                centers = (
                    0.5
                    * (lengths - radii)[:, None]
                    * self._batched_episode_rng.uniform(0, 0, size=(2,))
                )

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)
                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            if self.random_initial_position:
                xy = randomization.uniform(
                    low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self.peg_half_sizes[env_idx, 2]
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
                )
            else:
                pos = torch.zeros((b,3))
                pos[:,0],pos[:,1] = 0.0,-0.15
                pos[:, 2] = self.peg_half_sizes[env_idx, 2]
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(np.pi / 2, np.pi / 2),
                )

            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            if self.simple_target_position:
                # xy = randomization.uniform(
                #     low=torch.tensor([-0.1, 0.2]),
                #     high=torch.tensor([0.1, 0.4]),
                #     size=(b, 2),
                # )
                # xy[:,1] = 0.3
                xy_x_options = torch.tensor([-0.1, 0.1])
                xy_x = xy_x_options[torch.randint(0, len(xy_x_options), (b,))]

                # Pin xy[:, 1] to 0.3
                xy_y = torch.full((b,), 0.3)

                # Build xy
                xy = torch.stack((xy_x, xy_y), dim=1)

                # Initialize pos
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self.peg_half_sizes[env_idx, 0]
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(np.pi / 2 , np.pi / 2 ),
                )
            
            else:
                xy = randomization.uniform(
                    low=torch.tensor([-0.1, 0.2]),
                    high=torch.tensor([0.1, 0.4]),
                    size=(b, 2),
                )
                pos = torch.zeros((b, 3))
                pos[:, :2] = xy
                pos[:, 2] = self.peg_half_sizes[env_idx, 0]
                quat = randomization.random_quaternions(
                    b,
                    self.device,
                    lock_x=True,
                    lock_y=True,
                    bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs
    
    def _smooth_action_penalty(self,action):
        ## First Order
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action).to(self.device)
        if len(action.shape) < 2:
            action = action.reshape(1,-1)
        non_grasp_action = action[:,:-1]
        if len(self.previous_actions.shape) < 2:
            self.previous_actions = self.previous_actions.reshape(1,-1)
        prev_non_grasp_action = self.previous_actions[:,:-1]

        action_norm = torch.linalg.norm(non_grasp_action, axis = -1) ## in averge 1.4 

        ## Second Order 
        delta_action_norm = torch.linalg.norm( (non_grasp_action - prev_non_grasp_action), axis = -1 ) 

        ## initial version 0.1, 0.01 
        c1 = 0.0
        c2 = 0.0

        return action_norm * c1 + delta_action_norm * c2

    @property
    def is_grasp_peg(self):
        return self.agent.is_grasping(self.peg, max_angle=20)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose * (offset)
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        reward = reaching_reward + is_grasped

        # Stage 3: Orient the grasped peg properly towards the hole

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )

        # Stage 4: Insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5 * (
            1
            - torch.tanh(
                5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
            )
        )
        reward += insertion_reward * (is_grasped & pre_inserted)
        
        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10
    
    def _parse_obs(self,obs):
        agent_info = obs['agent']
        qpos,qvel = agent_info['qpos'],agent_info['qvel'] ### 7 + 7
        extra_info = obs['extra'] 
        tcp_pose = extra_info['tcp_pose'] ## 7 
        ### peg info 
        peg_pose = extra_info['peg_pose'] ## 7
        peg_half_size = extra_info['peg_half_size'] ## 3 
        box_hole_pose = extra_info['box_hole_pose']
        box_hole_radius = extra_info['box_hole_radius'].unsqueeze(-1)

        goal = torch.cat([box_hole_pose,box_hole_radius],dim=-1)
        state = torch.cat(
            [qpos,qvel,tcp_pose,peg_pose,peg_half_size], dim=-1
        )
        return state, goal 

    def set_penalty_clip(self,ratio):
        self.penalty_clip_ratio = ratio

    def step(self,act):
        obs,reward, terminated, trunctated, info = super().step(act)
        action_penalty = self._smooth_action_penalty(act)
        # clipped_action_penalty = torch.clamp(action_penalty, min = torch.zeros_like(action_penalty), max = reward * self.penalty_clip_ratio)
        # action_penalty = clipped_action_penalty

        # add penalty 
        info['raw_reward'] = reward.clone()
        reward -= action_penalty
        # record action 
        self.previous_actions = act 

        state,goal = self._parse_obs(obs)
        info['goal'] = goal
        info['penalty'] = action_penalty
        
        return state, reward, terminated,trunctated,info

    def get_state_info(self):
        obs = super().get_obs()
        state,goal = self._parse_obs(obs)
        info = {}
        info['goal'] = goal 
        return state, info 
    
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
        
        

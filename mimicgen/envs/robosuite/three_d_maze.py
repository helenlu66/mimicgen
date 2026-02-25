"""
Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the NVIDIA Source Code License [see LICENSE for details].

Contains 3D Maze environment for single arm manipulation.
"""
import numpy as np
from copy import deepcopy

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, add_material

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from mimicgen.models.robosuite.objects import BoxPatternObject


class ThreeDMazeEnv(SingleArmEnv_MG):
    """
    Single arm manipulation environment with a 3D maze on a table.
    The maze is constructed using a BoxPatternObject.
    """
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_offset=(0, 0, 0.8),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        **kwargs
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_offset = np.array(table_offset)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            **kwargs
        )

    def _load_model(self):
        """
        Loads arena, robot, and maze object.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        # Create material for maze
        maze_material = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        # Define maze pattern - 1 means there's a wall block, 0 means empty space
        # This creates a simple 3D maze structure
        self.maze_unit_size = 0.02
        self.maze_pattern = [
            # Layer 0 (bottom) - base walls
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            # Layer 1 (middle) - walls
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            # Layer 2 (top) - partial walls
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ]

        # Create the maze as a BoxPatternObject
        self.maze = BoxPatternObject(
            name="maze",
            unit_size=[self.maze_unit_size, self.maze_unit_size, self.maze_unit_size],
            pattern=self.maze_pattern,
            rgba=None,
            material=maze_material,
            density=100,
            friction=(1.0, 0.005, 0.0001),
        )

        # Get placement initializer
        self._get_placement_initializer()

        mujoco_objects = [self.maze]

        # Task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

        self.objects = [self.maze]

    def _get_placement_initializer(self):
        """
        Set up placement sampler for the maze on the table.
        """
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Maze placement
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-maze",
                mujoco_objects=self.maze,
                x_range=bounds["maze"]["x"],
                y_range=bounds["maze"]["y"],
                rotation=bounds["maze"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["maze"]["reference"],
                z_offset=0.10,
            )
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of maze.
        """
        return dict(
            maze=dict(
                x=(0.0, 0.0),  # Center of table
                y=(0.0, 0.0),  # Center of table
                z_rot=(0., 0.),  # No rotation
                reference=self.table_offset,
            ),
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Manually place maze on table after reset
        self._place_maze_on_table()

    def _place_maze_on_table(self):
        """
        Manually place the maze on top of the table.
        """
        # Get current maze position
        maze_pos = self.sim.data.body_xpos[self.maze_body_id].copy()
        maze_quat = self.sim.data.body_xquat[self.maze_body_id].copy()
        
        # Set position on table surface (table_offset[2] + half table height + maze height offset)
        maze_pos[2] = self.table_offset[2] + self.table_full_size[2]/2 + 0.03
        
        # Update maze position
        self.sim.data.set_joint_qpos(self.maze.joints[0], np.concatenate([maze_pos, maze_quat]))
        self.sim.forward()

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()
        self.maze_body_id = self.sim.model.body_name2id(self.maze.root_body)

    def _check_success(self):
        """
        Check if task is successful. Override in subclass for specific task completion criteria.
        """
        return False

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        reward = 0.0

        # Sparse reward structure - override in subclass for specific task
        if self._check_success():
            reward = 1.0

        return reward

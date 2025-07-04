# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial, add_material, find_elements, string_to_array

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

import mimicgen
import xml.etree.ElementTree as ET
from mimicgen.models.robosuite.objects import BlenderObject, CoffeeMachinePodObject, CoffeeMachineObject, LongDrawerObject, CupObject
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class Coffee_Pre_Novelty(SingleArmEnv_MG):
    """
    This class corresponds to the coffee task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
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
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

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
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        The sparse reward only consists of the threading component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        # initialize objects of interest
        self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        self.coffee_machine = CoffeeMachineObject(name="coffee_machine")
        self.coffee_machine_lid = self.coffee_machine.lid
        self.coffee_pod_holder = self.coffee_machine.pod_holder
        
        objects = [self.coffee_pod, self.coffee_machine]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            coffee_machine=dict(
                x=(0.0, 0.0),
                y=(-0.1, -0.1),
                z_rot=(-np.pi / 6., -np.pi / 6.),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.13, -0.07),
                y=(0.17, 0.23),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=bounds["coffee_machine"]["x"],
                y_range=bounds["coffee_machine"]["y"],
                rotation=bounds["coffee_machine"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_machine"]["reference"],
                z_offset=0.,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeePodSampler",
                mujoco_objects=self.coffee_pod,
                x_range=bounds["coffee_pod"]["x"],
                y_range=bounds["coffee_pod"]["y"],
                rotation=bounds["coffee_pod"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_pod"]["reference"],
                z_offset=0.,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references for this env
        self.obj_body_id = dict(
            coffee_pod=self.sim.model.body_name2id(self.coffee_pod.root_body),
            # coffee_machine=self.sim.model.body_name2id(self.coffee_machine.root_body), don't need coffee machine pos and angles for RL
            coffee_pod_holder=self.sim.model.body_name2id("coffee_machine_pod_holder_root"),
            coffee_machine_lid=self.sim.model.body_name2id("coffee_machine_lid_main"),
        )
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr("coffee_machine_lid_main_joint0")

        # for checking contact (used in reward function, and potentially observation space)
        self.pod_geom_id = self.sim.model.geom_name2id("coffee_pod_g0")
        self.lid_geom_id = self.sim.model.geom_name2id("coffee_machine_lid_g0")
        pod_holder_geom_names = ["coffee_machine_pod_holder_cup_body_hc_{}".format(i) for i in range(64)]
        self.pod_holder_geom_ids = [self.sim.model.geom_name2id(x) for x in pod_holder_geom_names]

        # size of bounding box for pod holder
        self.pod_holder_size = self.coffee_machine.pod_holder_size

        # size of bounding box for pod
        self.pod_size = self.coffee_pod.get_bounding_box_half_size()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Always reset the hinge joint position
        self.sim.data.qpos[self.hinge_qpos_addr] = 2. * np.pi / 3.
        self.sim.forward()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            actives = [False]

            @sensor(modality=modality)
            def eef_control_frame_pose(obs_cache):
                return T.make_pose(
                    np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.robots[0].controller.eef_name)]),
                    np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.robots[0].controller.eef_name)].reshape([3, 3])),
                ) if \
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors += [eef_control_frame_pose]
            names += ["eef_control_frame_pose"]
            actives += [False]

            # add ground-truth poses (absolute and relative to eef) for all objects
            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)


            # add hinge angle of lid
            @sensor(modality=modality)
            def hinge_angle(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])
            sensors += [hinge_angle]
            names += ["hinge_angle"]
            actives += [False]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_euler_angles(obs_cache):
            return T.mat2euler(T.quat2mat(T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")))
        
        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if "world_pose_in_gripper" not in obs_cache:
                return np.zeros(3)
            obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])
            obj_quat = T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")
            obj_pose = T.pose2mat((obj_pos, obj_quat))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            obs_cache[f"{obj_name}_pose"] = obj_pose
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_to_eef_pos]
        names = [f"{obj_name}_to_{pf}eef_pos"]

        return sensors, names

    def _create_obj_centric_sensors(self, modality="object_centric"):
        """
        Creates sensors for poses relative to certain objects. This is abstracted in a separate 
        function call so that we don't have local function naming collisions during 
        the _setup_observables() call.

        Args:
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        sensors = []
        names = []
        pf = self.robots[0].robot_model.naming_prefix

        # helper function for relative position sensors, to avoid code duplication
        def _pos_helper(obs_cache, obs_name, ref_name, quat_cache_name):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [obs_name, ref_name]]):
                return np.zeros(3)
            ref_pose = obs_cache[ref_name]
            obs_pose = obs_cache[obs_name]
            rel_pose = T.pose_in_A_to_pose_in_B(obs_pose, T.pose_inv(ref_pose))
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[quat_cache_name] = rel_quat
            return rel_pos

        # helper function for relative quaternion sensors, to avoid code duplication
        def _quat_helper(obs_cache, quat_cache_name):
            return obs_cache[quat_cache_name] if \
                quat_cache_name in obs_cache else np.zeros(4)

        # eef pose relative to ref object frames
        @sensor(modality=modality)
        def eef_pos_rel_pod(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="coffee_pod_pose",
                quat_cache_name="eef_quat_rel_pod",
            )
        @sensor(modality=modality)
        def eef_quat_rel_pod(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_pod",
            )
        sensors += [eef_pos_rel_pod, eef_quat_rel_pod]
        names += [f"{pf}eef_pos_rel_pod", f"{pf}eef_quat_rel_pod"]

        @sensor(modality=modality)
        def eef_pos_rel_pod_holder(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="coffee_pod_holder_pose",
                quat_cache_name="eef_quat_rel_pod_holder",
            )
        @sensor(modality=modality)
        def eef_quat_rel_pod_holder(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_pod_holder",
            )
        sensors += [eef_pos_rel_pod_holder, eef_quat_rel_pod_holder]
        names += [f"{pf}eef_pos_rel_pod_holder", f"{pf}eef_quat_rel_pod_holder"]

        # obj pose relative to ref object frame
        @sensor(modality=modality)
        def pod_pos_rel_pod_holder(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="coffee_pod_pose",
                ref_name="coffee_pod_holder_pose",
                quat_cache_name="pod_quat_rel_pod_holder",
            )
        @sensor(modality=modality)
        def pod_quat_rel_pod_holder(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="pod_quat_rel_pod_holder",
            )
        sensors += [pod_pos_rel_pod_holder, pod_quat_rel_pod_holder]
        names += ["pod_pos_rel_pod_holder", "pod_quat_rel_pod_holder"]

        return sensors, names

    def _check_success(self):
        """
        Check if task is complete.
        """
        metrics = self._get_partial_task_metrics()
        return metrics["task"]

    def _check_lid(self):
        # lid should be closed (angle should be less than 5 degrees)
        hinge_tolerance = 15. * np.pi / 180. 
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        lid_check = (hinge_angle < hinge_tolerance)
        return lid_check
    
    def check_can_flip_up_lid(self):
        """
        Returns True if the coffee pod lid can be flipped up.
        """
        hinge_tolerance = 90. * np.pi / 180. 
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        lid_check = (hinge_angle < hinge_tolerance)
        return lid_check
    
    def check_directly_on_table(self, obj_name):
        """
        Returns True if the object is directly on the table.
        """
        # if obj_name == 'coffee-pod':
        #     obj_name = 'coffee_pod'
        # if hasattr(self, obj_name):
        #     obj_bounding_box = getattr(self, obj_name).get_bounding_box_half_size()
        # elif obj_name == 'coffee-machine-lid':
        #     obj_name = 'coffee_machine_lid'
        #     obj_bounding_box = self.coffee_machine.lid_size
        # elif obj_name == 'coffee-pod-holder':
        #     obj_name = 'coffee_pod_holder'
        #     obj_bounding_box = self.pod_holder_size
        # elif obj_name == 'drawer':
        #     obj_bounding_box = self.cabinet_object.get_bounding_box_half_size()
        # table_z_offset = self.table_offset[2]
        # obj_z = self.sim.data.body_xpos[self.obj_body_id[obj_name]][2]
        # obj_bottom_z = obj_z - obj_bounding_box[2]
        # return obj_bottom_z - table_z_offset < 0.01
        obj_name = obj_name.replace('-', '_')
        obj = getattr(self, obj_name)
        return self.check_contact(obj, 'table_collision')
        
    

    def check_pod(self):
        # pod should be in pod holder
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])
        pod_check = True
        pod_horz_check = True

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        if np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) > r_diff:
            pod_check = False
            pod_horz_check = False

        # make sure vertical pod dimension is above pod holder lower bound and below the lid lower bound
        lid_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_machine_lid"]])
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        z_lim_high = lid_pos[2] - self.coffee_machine.lid_size[2]
        if (pod_pos[2] - self.pod_size[2] < z_lim_low) or (pod_pos[2] + self.pod_size[2] > z_lim_high):
            pod_check = False
        return pod_check

    def _get_partial_task_metrics(self):
        metrics = dict()

        lid_check = self._check_lid()
        pod_check = self.check_pod()

        # pod should be in pod holder
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])
        pod_horz_check = True

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        if np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) > r_diff:
            pod_horz_check = False

        # make sure vertical pod dimension is above pod holder lower bound and below the lid lower bound
        lid_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_machine_lid"]])
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]

        metrics["task"] = lid_check and pod_check

        # for pod insertion check, just check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance)
        metrics["insertion"] = pod_horz_check and pod_z_check

        # pod grasp check
        metrics["grasp"] = self._check_pod_is_grasped()

        # check is True if the pod is on / near the rim of the pod holder
        rim_horz_tolerance = 0.03
        rim_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance)

        rim_vert_tolerance = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance) and (rim_vert_length > 0.)
        metrics["rim"] = rim_horz_check and rim_vert_check

        return metrics

    def _check_pod_is_grasped(self):
        """
        check if pod is grasped by robot
        """
        return self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.coffee_pod.contact_geoms]
        )

    def _check_pod_and_pod_holder_contact(self):
        """
        check if pod is in contact with the container
        """
        pod_and_pod_holder_contact = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if(
                ((contact.geom1 == self.pod_geom_id) and (contact.geom2 in self.pod_holder_geom_ids)) or
                ((contact.geom2 == self.pod_geom_id) and (contact.geom1 in self.pod_holder_geom_ids))
            ):
                pod_and_pod_holder_contact = True
                break
        return pod_and_pod_holder_contact

    def _check_pod_on_rim(self):
        """
        check if pod is on pod container rim and not being inserted properly (for reward check)
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        # check if pod is in contact with the container
        pod_and_pod_holder_contact = self._check_pod_and_pod_holder_contact()

        # check that pod vertical position is not too low or too high
        rim_vert_tolerance_1 = 0.022
        rim_vert_tolerance_2 = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length > rim_vert_tolerance_1) and (rim_vert_length < rim_vert_tolerance_2)

        return (pod_and_pod_holder_contact and rim_vert_check)

    def _check_pod_being_inserted(self):
        """
        check if robot is in the process of inserting the pod into the container
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        rim_horz_tolerance = 0.005
        rim_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance)

        rim_vert_tolerance_1 = -0.01
        rim_vert_tolerance_2 = 0.023
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance_2) and (rim_vert_length > rim_vert_tolerance_1)

        return (rim_horz_check and rim_vert_check)

    def _check_pod_inserted(self):
        """
        check if pod has been inserted successfully
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        pod_horz_check = True
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        pod_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) <= r_diff)

        # check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance)
        return (pod_horz_check and pod_z_check)

    def _check_lid_being_closed(self):
        """
        check if lid is being closed
        """

        # (check for hinge angle being less than default angle value, 120 degrees)
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        return (hinge_angle < 2.09)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the coffee machine.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the coffee machine
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.coffee_machine)


class Coffee_D0(Coffee_Pre_Novelty):
    """Rename base class for convenience."""
    pass


class Coffee_D1(Coffee_D0):
    """
    Wider initialization for pod and coffee machine.
    """
    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            coffee_machine=dict(
                x=(0.05, 0.15),
                y=(-0.2, -0.1),
                z_rot=(-np.pi / 6., np.pi / 3.),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                # x=(-0.2, -0.2),
                x=(-0.2, 0.05),
                # x=(-0.13, -0.07),
                y=(0.17, 0.3),
                # y=(0.3, 0.3),
                # y=(0.1, 0.1),
                # y=(0.17, 0.23),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )


class Coffee_D2(Coffee_D1):
    """
    Similar to Coffee_D1, but put pod on the left, and machine on the right. Had to also move
    machine closer to robot (in x) to get kinematics to work out.
    """
    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            coffee_machine=dict(
                x=(-0.05, 0.05),
                y=(0.1, 0.2),
                z_rot=(2. * np.pi / 3., 7. * np.pi / 6.),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.2, 0.05),
                y=(-0.3, -0.17),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )


class Coffee_Drawer_Novelty(Coffee_Pre_Novelty):
    """
    Harder coffee task where the task starts with materials in drawer and coffee machine closed. The robot
    needs to retrieve the coffee pod and mug from the drawer, open the coffee machine, place the pod and mug 
    in the machine, and then close the lid.
    """
    def _get_mug_model(self):
        """
        Allow subclasses to override which mug to use.
        """
        shapenet_id = "3143a4ac" # beige round mug, works well and matches color scheme of other assets
        shapenet_scale = 1.0
        base_mjcf_path = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/shapenet_core/mugs")
        mjcf_path = os.path.join(base_mjcf_path, "{}/model.xml".format(shapenet_id))

        self.mug = BlenderObject(
            name="mug",
            mjcf_path=mjcf_path,
            scale=shapenet_scale,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1, 1, 1),
            margin=0.001,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        # Create drawer object
        tex_attrib = {
            "type": "cube"
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        self.drawer = LongDrawerObject(name="DrawerObject")

        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)
        obj_body = self.drawer
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                             naming_prefix=obj_body.naming_prefix,
                                                             custom_material=deepcopy(material))
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)

        # Create mug
        self._get_mug_model()

        # Create coffee pod and machine (note that machine no longer has a cup!)
        self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        self.coffee_machine = CoffeeMachineObject(name="coffee_machine", add_cup=False)
        self.coffee_pod_holder = self.coffee_machine.pod_holder
        self.coffee_machine_lid = self.coffee_machine.lid
        # objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object, self.mug]
        objects = [self.coffee_pod, self.coffee_machine, self.drawer]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )
        # HACK: merge in mug afterwards because its number of geoms may change
        #       and this may break the generate_id_mappings function in task.py
        self.model.merge_objects([self.mug]) # add cleanup object to model 

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            drawer=dict(
                # x=(0.1, 0.1),
                # y=(0.3, 0.3),
                # z_rot=(0.0, 0.0),
                # x=(0.15, 0.15),
                x=(-0.02, -0.02),
                y=(-0.35, -0.35),
                z_rot=(np.pi, np.pi),
                reference=self.table_offset,
            ),
            coffee_machine=dict(
                x = (-0.3, -0.3),
                y = (-0.22, -0.22),
                # x=(-0.15, -0.15),
                # y=(-0.25, -0.25),
                z_rot=(0, 0),
                # put vertical
                # z_rot=(-np.pi / 2., -np.pi / 2.),
                reference=self.table_offset,
            ),
            mug=dict(
                # upper right
                # x=(-0.2, -0.2),
                # y=(0.17, 0.23),
                # z_rot=(0.0, 0.0),
                # lower right
                # x=(0.05, 0.20),
                # y=(-0.25, -0.05),
                # z_rot=(0.0, 0.0),
                # middle
                x=(-0.275, -0.275),
                # y=(0, 0),
                y=(-0.095, -0.095),
                # y=(-0.082, -0.082),
                z_rot=(0.0, 0.0), 
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                # x=(-0.032, 0.032),
                x=(-0.02, 0.02),
                y=(-0.02, 0.00),
                z_rot=(0.0, 0.0),
                reference=np.array((0., 0., 0.)),
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="DrawerSampler",
                mujoco_objects=self.drawer,
                x_range=bounds["drawer"]["x"],
                y_range=bounds["drawer"]["y"],
                rotation=bounds["drawer"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["drawer"]["reference"],
                z_offset=0.03,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=bounds["coffee_machine"]["x"],
                y_range=bounds["coffee_machine"]["y"],
                rotation=bounds["coffee_machine"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_machine"]["reference"],
                z_offset=0.,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="MugSampler",
                mujoco_objects=self.mug,
                x_range=bounds["mug"]["x"],
                y_range=bounds["mug"]["y"],
                rotation=bounds["mug"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=bounds["mug"]["reference"],
                z_offset=0.01,
            )
        )

        # Note: coffee pod gets its own placement sampler to sample within a box within the drawer.

        # First, got drawer geom size with self.sim.model.geom_size[self.drawer_bottom_geom_id]
        # Value is array([0.08 , 0.09 , 0.008])
        # Then, used this to set reasonable box for pod init within drawer.
        self.pod_placement_initializer = UniformRandomSampler(
            name="CoffeePodInDrawerSampler",
            mujoco_objects=self.coffee_pod,
            x_range=bounds["coffee_pod"]["x"],
            y_range=bounds["coffee_pod"]["y"],
            rotation=bounds["coffee_pod"]["z_rot"],
            rotation_axis='z',
            # ensure_object_boundary_in_range=True, # make sure pod fits within the box
            ensure_object_boundary_in_range=False, # make sure pod fits within the box
            ensure_valid_placement=True,
            reference_pos=bounds["coffee_pod"]["reference"],
            z_offset=0.,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.cabinet_qpos_addr = self.sim.model.get_joint_qpos_addr(self.drawer.joints[0])
        self.obj_body_id["drawer"] = self.sim.model.body_name2id(self.drawer.root_body)
        self.obj_body_id["mug"] = self.sim.model.body_name2id(self.mug.root_body)
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id("DrawerObject_drawer_bottom")
        self.drawer_handle_mid_geom_id = self.sim.model.geom_name2id("DrawerObject_drawer_handle_2")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.drawer:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    # obj_pos_to_set[2] = 0.905 # hardcode z-value to correspond to parent class
                    obj_pos_to_set[2] = 0.805 # hardcode z-value to make sure it lies on table surface
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Always reset the hinge joint position, and the cabinet slide joint position

        # Hinge for coffee machine starts closed
        self.sim.data.qpos[self.hinge_qpos_addr] = 0.
        # self.sim.data.qpos[self.hinge_qpos_addr] = 2. * np.pi / 3.

        # Cabinet should start closed (0.) but can set to open (-0.135) for debugging.
        self.sim.data.qpos[self.cabinet_qpos_addr] = 0.
        # self.sim.data.qpos[self.cabinet_qpos_addr] = -0.135
        self.sim.forward()

        if not self.deterministic_reset:

            # sample pod location relative to center of drawer bottom geom surface
            coffee_pod_placement = self.pod_placement_initializer.sample(on_top=False)
            assert len(coffee_pod_placement) == 1
            rel_pod_pos, rel_pod_quat, pod_obj = list(coffee_pod_placement.values())[0]
            rel_pod_pos, rel_pod_quat = np.array(rel_pod_pos), np.array(rel_pod_quat)
            assert pod_obj is self.coffee_pod

            # center of drawer bottom
            drawer_bottom_geom_pos = np.array(self.sim.data.geom_xpos[self.drawer_bottom_geom_id])

            # our x-y relative position is sampled with respect to drawer geom frame. Here, we use the drawer's rotation
            # matrix to convert this relative position to a world relative position, so we can add it to the drawer world position
            drawer_rot_mat = T.quat2mat(T.convert_quat(self.sim.model.body_quat[self.sim.model.body_name2id(self.drawer.root_body)], to="xyzw"))
            rel_pod_pos[:2] = drawer_rot_mat[:2, :2].dot(rel_pod_pos[:2])

            # also convert the sampled pod rotation to world frame
            rel_pod_mat = T.quat2mat(T.convert_quat(rel_pod_quat, to="xyzw"))
            pod_mat = drawer_rot_mat.dot(rel_pod_mat)
            pod_quat = T.convert_quat(T.mat2quat(pod_mat), to="wxyz")

            # get half-sizes of drawer geom and coffee pod to place coffee pod at correct z-location (on top of drawer bottom geom)
            drawer_bottom_geom_z_offset = self.sim.model.geom_size[self.drawer_bottom_geom_id][-1] # half-size of geom in z-direction
            coffee_pod_bottom_offset = np.abs(self.coffee_pod.bottom_offset[-1])
            coffee_pod_z = drawer_bottom_geom_pos[2] + drawer_bottom_geom_z_offset + coffee_pod_bottom_offset + 0.001

            # set coffee pod in center of drawer
            pod_pos = np.array(drawer_bottom_geom_pos) + rel_pod_pos
            pod_pos[-1] = coffee_pod_z

            self.sim.data.set_joint_qpos(pod_obj.joints[0], np.concatenate([np.array(pod_pos), np.array(pod_quat)]))


    def _reset_open_drawer_open_lid(self):
        """
        Reset the environment with the drawer open and the coffee machine lid open.
        """
        SingleArmEnv._reset_internal(self)
        
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.drawer:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    # obj_pos_to_set[2] = 0.905 # hardcode z-value to correspond to parent class
                    obj_pos_to_set[2] = 0.805 # hardcode z-value to make sure it lies on table surface
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
        self.sim.data.qpos[self.cabinet_qpos_addr] = -0.195
        self.sim.data.qpos[self.hinge_qpos_addr] = 2. * np.pi / 3.
        self.sim.forward()

        if not self.deterministic_reset:

            # sample pod location relative to center of drawer bottom geom surface
            coffee_pod_placement = self.pod_placement_initializer.sample(on_top=False)
            assert len(coffee_pod_placement) == 1
            rel_pod_pos, rel_pod_quat, pod_obj = list(coffee_pod_placement.values())[0]
            rel_pod_pos, rel_pod_quat = np.array(rel_pod_pos), np.array(rel_pod_quat)
            assert pod_obj is self.coffee_pod

            # center of drawer bottom
            drawer_bottom_geom_pos = np.array(self.sim.data.geom_xpos[self.drawer_bottom_geom_id])

            # our x-y relative position is sampled with respect to drawer geom frame. Here, we use the drawer's rotation
            # matrix to convert this relative position to a world relative position, so we can add it to the drawer world position
            drawer_rot_mat = T.quat2mat(T.convert_quat(self.sim.model.body_quat[self.sim.model.body_name2id(self.drawer.root_body)], to="xyzw"))
            rel_pod_pos[:2] = drawer_rot_mat[:2, :2].dot(rel_pod_pos[:2])

            # also convert the sampled pod rotation to world frame
            rel_pod_mat = T.quat2mat(T.convert_quat(rel_pod_quat, to="xyzw"))
            pod_mat = drawer_rot_mat.dot(rel_pod_mat)
            pod_quat = T.convert_quat(T.mat2quat(pod_mat), to="wxyz")

            # get half-sizes of drawer geom and coffee pod to place coffee pod at correct z-location (on top of drawer bottom geom)
            drawer_bottom_geom_z_offset = self.sim.model.geom_size[self.drawer_bottom_geom_id][-1] # half-size of geom in z-direction
            coffee_pod_bottom_offset = np.abs(self.coffee_pod.bottom_offset[-1])
            coffee_pod_z = drawer_bottom_geom_pos[2] + drawer_bottom_geom_z_offset + coffee_pod_bottom_offset + 0.001

            # set coffee pod in center of drawer
            pod_pos = np.array(drawer_bottom_geom_pos) + rel_pod_pos
            pod_pos[-1] = coffee_pod_z

            self.sim.data.set_joint_qpos(pod_obj.joints[0], np.concatenate([np.array(pod_pos), np.array(pod_quat)]))


    def rename_observables(self, observables):
        """
        Make copies of the observables with names that match the planning domain.
        """
        # Positions of objects relative to the eef
        observables['drawer1_cabinet_to_gripper1_pos'] = observables.pop('drawer_to_robot0_eef_pos')
        observables['drawer1_handle_to_gripper1_pos'] = observables.pop('drawer_handle_to_robot0_eef_pos')
        observables['coffee-pod1_to_gripper1_pos'] = observables.pop('coffee_pod_to_robot0_eef_pos')
        observables['coffee-pod-holder1_to_gripper1_pos'] = observables.pop('coffee_pod_holder_to_robot0_eef_pos')
        observables['mug1_to_gripper1_pos'] = observables.pop('mug_to_robot0_eef_pos')
        observables['lid1_to_gripper1_pos'] = observables.pop('coffee_machine_lid_to_robot0_eef_pos')
        observables['table1_center_to_gripper1_pos'] = observables.pop('table_center_to_robot0_eef_pos')

        # Positions of the eef
        observables['gripper1_pos'] = observables.pop('robot0_eef_pos')
        observables['gripper1_aperture'] = observables.pop('robot0_gripper_aperture')
        observables['griper1_fingers_qpos'] = observables.pop('robot0_gripper_fingers_qpos')
        
        # # Positions of objects
        # observables['gripper1_pos'] = observables.pop('robot0_eef_pos')
        # observables['griper1_fingers_qpos'] = observables.pop('robot0_gripper_fingers_qpos')
        # observables['drawer1_cabinet_pos'] = observables.pop('drawer_pos')
        # observables['drawer1_handle_pos'] = observables.pop('drawer_handle_pos')
        # observables['coffee-pod1_pos'] = observables.pop('coffee_pod_pos')
        # observables['coffee-pod-holder1_pos'] = observables.pop('coffee_pod_holder_pos')
        # observables['mug1_pos'] = observables.pop('mug_pos')
        # observables['lid1_pos'] = observables.pop('coffee_machine_lid_pos')
        # observables['table1_pos'] = observables.pop('table_pos')
        
        # # Orientations of objects
        # observables['gripper1_euler_angles'] = observables.pop('robot0_eef_euler_angles')
        # observables['drawer1_cabinet_euler_angles'] = observables.pop('drawer_euler_angles')
        # observables['drawer1_handle_euler_angles'] = observables.pop('drawer_handle_euler_angles')
        # observables['coffee-pod1_euler_angles'] = observables.pop('coffee_pod_euler_angles')
        # observables['coffee-pod_holder1_euler_angles'] = observables.pop('coffee_pod_holder_euler_angles')
        # observables['mug1_euler_angles'] = observables.pop('mug_euler_angles')
        # observables['lid1_euler_angles'] = observables.pop('coffee_machine_lid_euler_angles')
        # observables['table1_euler_angles'] = observables.pop('table_euler_angles')


        return observables
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """

        # super class will populate observables for "mug" and "drawer" poses in addition to coffee machine and coffee pod
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        if self.use_object_obs:
            modality = "object"

            # add drawer joint angle observable
            @sensor(modality=modality)
            def drawer_joint_angle(obs_cache):
                return np.array([self.sim.data.qpos[self.cabinet_qpos_addr]])

            @sensor(modality=modality)
            def drawer_handle_pos(obs_cache):
                return np.array(self.sim.data.geom_xpos[self.drawer_handle_mid_geom_id])


            # relative position of drawer handle
            @sensor(modality=modality)
            def drawer_handle_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if "world_pose_in_gripper" not in obs_cache:
                    return np.zeros(3)
                drawer_handle_pos = np.array(self.sim.data.geom_xpos[self.drawer_handle_mid_geom_id])
                drawer_handle_quat = np.array(self.sim.data.geom_xmat[self.drawer_handle_mid_geom_id])
                drawer_handle_pose = T.pose2mat((drawer_handle_pos, drawer_handle_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(drawer_handle_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"drawer_handle_to_{pf}eef_quat"] = rel_quat
                obs_cache[f"drawer_handle_to_{pf}eef_pos"] = rel_pos
                return rel_pos


            @sensor(modality=modality)
            def drawer_handle_euler_angles(obs_cache):
                return T.mat2euler(T.quat2mat(self.sim.data.geom_xmat[self.drawer_handle_mid_geom_id]).reshape(3, 3))


            @sensor(modality=modality)
            def table_pos(obs_cache):
                return np.array(self.table_offset)
            
            @sensor(modality=modality)
            def table_center_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if "world_pose_in_gripper" not in obs_cache:
                    return np.zeros(3)
                table_pos = np.array(self.table_offset)
                table_quat = np.array([0, 0, 0, 1])
                table_pose = T.pose2mat((table_pos, table_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(table_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"table_center_to_{pf}eef_quat"] = rel_quat
                obs_cache[f"table_center_to_{pf}eef_pos"] = rel_pos
                return rel_pos


            @sensor(modality=modality)
            def table_euler_angles(obs_cache):
                return np.array([0, 0, 0])


            names = [f"drawer_handle_to_{pf}eef_pos", f"table_center_to_{pf}eef_pos"]
            sensors = [drawer_handle_to_eef_pos, table_center_to_eef_pos]
            actives = [True, True]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )
        
        observables = self.rename_observables(observables)

        return observables
    
    def _create_overlap_sensors(self):
        """
        Create sensors for the overlap between objects.
        """
        sensors = []
        names = []

        modality = "object"

        # mug1 overlaps
        mug1_drawer1_overlap = self.estimate_obj1_overlap_w_obj2("mug", "drawer")
        mug1_coffee_pod_holder1_overlap = self.estimate_obj1_overlap_w_obj2("mug", "coffee_pod_holder")

        # coffee pod overlaps
        coffee_pod1_drawer1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_pod", "drawer")
        coffee_pod1_mug1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_pod", "mug")
        coffee_pod1_coffee_pod_holder1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_pod", "coffee_pod_holder")

        # coffee machine lid overlaps
        coffee_machine_lid1_drawer1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_machine_lid", "drawer")
        coffee_machine_lid1_mug1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_machine_lid", "mug")
        coffee_machine_lid1_coffee_pod_holder1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_machine_lid", "coffee_pod_holder")

        # coffee pod holder overlaps
        coffee_pod_holder1_drawer1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_pod_holder", "drawer")
        coffee_pod_holder1_mug1_overlap = self.estimate_obj1_overlap_w_obj2("coffee_pod_holder", "mug")

        # add sensors
        @sensor(modality=modality)
        def percent_overlap_of_mug1_with_drawer1(obs_cache):
            return np.array([mug1_drawer1_overlap])
        sensors += [percent_overlap_of_mug1_with_drawer1]
        names += ["percent_overlap_of_mug1_with_drawer1"]

        @sensor(modality=modality)
        def percent_overlap_of_mug1_with_coffee_pod_holder1(obs_cache):
            return np.array([mug1_coffee_pod_holder1_overlap])
        sensors += [percent_overlap_of_mug1_with_coffee_pod_holder1]
        names += ["percent_overlap_of_mug1_with_coffee_pod_holder1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod1_with_drawer1(obs_cache):
            return np.array([coffee_pod1_drawer1_overlap])
        sensors += [percent_overlap_of_coffee_pod1_with_drawer1]
        names += ["percent_overlap_of_coffee-pod1_with_drawer1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod1_with_mug1(obs_cache):
            return np.array([coffee_pod1_mug1_overlap])
        sensors += [percent_overlap_of_coffee_pod1_with_mug1]
        names += ["percent_overlap_of_coffee-pod1_with_mug1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod1_with_coffee_pod_holder1(obs_cache):
            return np.array([coffee_pod1_coffee_pod_holder1_overlap])
        sensors += [percent_overlap_of_coffee_pod1_with_coffee_pod_holder1]
        names += ["percent_overlap_of_coffee-pod1_with_coffee_pod_holder1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_machine_lid1_with_drawer1(obs_cache):
            return np.array([coffee_machine_lid1_drawer1_overlap])
        sensors += [percent_overlap_of_coffee_machine_lid1_with_drawer1]
        names += ["percent_overlap_of_coffee_machine_lid1_with_drawer1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_machine_lid1_with_mug1(obs_cache):
            return np.array([coffee_machine_lid1_mug1_overlap])
        sensors += [percent_overlap_of_coffee_machine_lid1_with_mug1]
        names += ["percent_overlap_of_coffee_machine_lid1_with_mug1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_machine_lid1_with_coffee_pod_holder1(obs_cache):
            return np.array([coffee_machine_lid1_coffee_pod_holder1_overlap])
        sensors += [percent_overlap_of_coffee_machine_lid1_with_coffee_pod_holder1]
        names += ["percent_overlap_of_coffee_machine_lid1_with_coffee_pod_holder1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod_holder1_with_drawer1(obs_cache):
            return np.array([coffee_pod_holder1_drawer1_overlap])
        sensors += [percent_overlap_of_coffee_pod_holder1_with_drawer1]
        names += ["percent_overlap_of_coffee_pod_holder1_with_drawer1"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod_holder1_with_mug1(obs_cache):
            return np.array([coffee_pod_holder1_mug1_overlap])
        sensors += [percent_overlap_of_coffee_pod_holder1_with_mug1]
        names += ["percent_overlap_of_coffee_pod_holder1_with_mug1"]

        return sensors, names

    def check_in_mug(self, obj_name):
        """
        Returns true if object is in the mug.
        """
        obj_name = obj_name.replace('-', '_')
        if obj_name != 'coffee_pod':
            return False # only coffee pod can be in the mug
        # get mug's half bounding box and pos
        # mug_half_bounding_box = self.mug.get_bounding_box_half_size()
        # mug_pos = self.sim.data.body_xpos[self.obj_body_id["mug"]]
        # # get object's bounding box and pos
        # obj_half_bounding_box = getattr(self, obj_name).get_bounding_box_half_size()
        # obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
        # # check if object is in mug
        # in_mug = np.all(np.abs(mug_pos - obj_pos) <= mug_half_bounding_box - obj_half_bounding_box)
        # return in_mug
        percent_overlap = self.estimate_obj1_overlap_w_obj2(obj_name, "mug")
        return percent_overlap > 0.5 # obj is considered in the mug if 50% of it is inside the mug
    
    def check_in_drawer(self, obj_name):
        """
        Returns true if object is in the drawer.
        """
        # check if object is in contact with the inside of the drawer
        assert obj_name in ["mug", "coffee-pod"]
        obj_name = obj_name.replace('-', '_')
        drawer_bottom_geom = "DrawerObject_drawer_bottom"
        contact = self.check_contact(drawer_bottom_geom, getattr(self, obj_name))
        if contact:
            return True
        else:
            percent_overlap = self.estimate_obj1_overlap_w_obj2(obj_name, "drawer")
            return percent_overlap > 0.1 # obj is considered in the drawer if more than 10% of it is inside the drawer 

    
    def check_drawer_open(self):
        """
        Returns true if drawer is open.
        """
        return (self.sim.data.qpos[self.cabinet_qpos_addr] <= -0.17)
    
    def check_drawer_open_percentage(self) -> float:
        """Returns how open the drawer is as a percentage.

        Returns:
            float: Percentage of how open the drawer is (0.0 to 1.0).
        """
        closed_joint_angle = 0.0
        open_joint_angle = -0.17
        joint_angle_range = closed_joint_angle - open_joint_angle # 0.17
        joint_angle = self.sim.data.qpos[self.cabinet_qpos_addr] # might be more negative than -0.17
        if joint_angle < open_joint_angle:
            return 1.0
        elif joint_angle > closed_joint_angle:
            return 0.0
        return -(open_joint_angle - joint_angle) / joint_angle_range

    
    def check_mug_upright(self):
        """Returns true if mug is upright. """
        # get the euler angles of the mug
        mug_euler_angles = T.mat2euler(T.quat2mat(T.convert_quat(self.sim.data.body_xquat[self.obj_body_id["mug"]], to="xyzw")))
        # check if the mug is upright i.e. roll and pitch angles are smaller than 45
        return np.all(np.abs(mug_euler_angles[:2]) < np.pi/4)
    
    def check_mug_under_pod_holder(self):
        """
        Returns true if mug is under the pod holder.
        """
        # check that the mug is making contact with the coffee machine base plate
        coffee_base_plate_geom = "coffee_machine_base_g0"
        mug_on_machine = self.check_contact(coffee_base_plate_geom, self.mug)
        return mug_on_machine
    
    def check_mug_placement(self):
        """
        Returns true if mug has been placed successfully on the coffee machine.
        """

        # check z-axis alignment by checking z unit-vector of obj pose and dot with (0, 0, 1)
        # then take cosine dist (1 - dot-prod)
        mug_upright = self.check_mug_upright()

        # to check if mug is placed on the machine successfully, we check that the mug is upright, and that it is
        # making contact with the coffee machine base plate
        mug_on_machine = self.check_mug_under_pod_holder()

        return mug_upright and mug_on_machine
    
    def estimate_obj1_overlap_w_obj2(self, obj1_name, obj2_name):
        """Estimate the percent overlap between two objects' bounding boxes. For example, if obj1 is fully inside obj2, the percentage is 1.

        Args:
            obj1_name (str): name of the first object
            obj2_name (str): name of the second object
        Returns:
            float: percentage of obj1's bounding box that overlaps with obj2's bounding box in [0, 1]
        """
        obj1 = getattr(self, obj1_name)
        if obj1_name in ('coffee_pod_holder', 'coffee_machine_lid'):
            obj1_id = self.sim.model.body_name2id("coffee_machine_" + obj1.root_body)
        else:
            obj1_id = self.sim.model.body_name2id(obj1.root_body)
        obj1_pos = self.sim.data.body_xpos[obj1_id]
        obj1_quat = self.sim.data.body_xquat[obj1_id]
        if obj1_name == "drawer":
            drawer_id = self.sim.model.body_name2id("DrawerObject_drawer_link")
            obj2_pos = self.sim.data.body_xpos[drawer_id]
            obj2_quat = self.sim.data.body_xquat[drawer_id]
            XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")
            xml_path = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer_long.xml")
            body_name = "drawer_link"  
            # Get bounding box dimensions
            length, width, height = self.get_bounding_box_dimensions(xml_path, body_name)
            obj1_half_bounding_box = np.array([length / 2, width / 2, height / 2])
        else:
            obj1_half_bounding_box = obj1.get_bounding_box_half_size()

        obj2 = getattr(self, obj2_name)
        if obj2_name in ('coffee_pod_holder', 'coffee_machine_lid'):
            obj2_id = self.sim.model.body_name2id("coffee_machine_" + obj2.root_body)
        else:
            obj2_id = self.sim.model.body_name2id(obj2.root_body)
        obj2_pos = self.sim.data.body_xpos[obj2_id]
        obj2_quat = self.sim.data.body_xquat[obj2_id]
        if obj2_name == "drawer":
            drawer_id = self.sim.model.body_name2id("DrawerObject_drawer_link")
            obj2_pos = self.sim.data.body_xpos[drawer_id]
            obj2_quat = self.sim.data.body_xquat[drawer_id]
            XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")
            xml_path = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer_long.xml")
            body_name = "drawer_link"  
            # Get bounding box dimensions
            length, width, height = self.get_bounding_box_dimensions(xml_path, body_name)
            obj2_half_bounding_box = np.array([length / 2, width / 2, height / 2])
        else:
            obj2_half_bounding_box = obj2.get_bounding_box_half_size()
    
        
        # find the min and max of obj1's bounding box in the local frame of obj2
        obj1_bounding_box_in_obj2_frame = self.local_frame_bounding_box(obj2_pos, obj2_quat, obj1_pos, obj1_half_bounding_box, obj1_quat)
        # find the min and max of obj2's bounding box
        obj2_bounding_box = (-obj2_half_bounding_box[0], +obj2_half_bounding_box[0], -obj2_half_bounding_box[1], +obj2_half_bounding_box[1], -obj2_half_bounding_box[2], +obj2_half_bounding_box[2])
        # calculate the overlap percentage. The bounding boxes are in the local frame of obj2
        overlap_x = max(0, min(obj1_bounding_box_in_obj2_frame[1], obj2_bounding_box[1]) - max(obj1_bounding_box_in_obj2_frame[0], obj2_bounding_box[0]))
        overlap_y = max(0, min(obj1_bounding_box_in_obj2_frame[3], obj2_bounding_box[3]) - max(obj1_bounding_box_in_obj2_frame[2], obj2_bounding_box[2]))
        overlap_z = max(0, min(obj1_bounding_box_in_obj2_frame[5], obj2_bounding_box[5]) - max(obj1_bounding_box_in_obj2_frame[4], obj2_bounding_box[4]))
        overlap_volume = overlap_x * overlap_y * overlap_z
        obj1_volume = (obj1_bounding_box_in_obj2_frame[1] - obj1_bounding_box_in_obj2_frame[0]) * (obj1_bounding_box_in_obj2_frame[3] - obj1_bounding_box_in_obj2_frame[2]) * (obj1_bounding_box_in_obj2_frame[5] - obj1_bounding_box_in_obj2_frame[4])
        return overlap_volume / obj1_volume
    
    def get_bounding_box_dimensions(self, xml_path: str, body_name: str):
        """
        Computes the bounding box dimensions (length, width, height) for a specific body in a Mujoco XML.
        
        Args:
        - xml_path (str): Path to the Mujoco XML file.
        - body_name (str): Name of the body whose bounding box dimensions are to be calculated.

        Returns:
        - tuple: A tuple containing (length, width, height)
        """
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all geoms in the specified body
        geoms = root.findall(f".//body[@name='{body_name}']//geom")

        # Initialize bounding box extents
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

        # Compute bounding box for each geom
        for geom in geoms:
            pos = list(map(float, geom.get('pos').split()))  # [x, y, z]
            size = list(map(float, geom.get('size').split()))
            geom_type = geom.get('type', 'box')  # Default to 'box' if type is not specified

            if geom_type == 'box':
                # Box has 3 size elements: [x, y, z]
                dx, dy, dz = size[0], size[1], size[2]
            elif geom_type in ['capsule', 'cylinder']:
                # Capsule/Cylinder have 2 size elements: [radius, half-length]
                dx = size[1]  # length of the capsule/cylinder
                dy = dz = size[0]  # radius determines the y and z size
            elif geom_type == 'sphere':
                # Sphere has 1 size element: [radius]
                dx = dy = dz = size[0]
            elif geom_type == 'plane':
                # Plane has 2 size elements: [x, y] (z is typically flat)
                dx = size[0]
                dy = size[1]
                dz = 0  # Planes don't have height (they're flat)
            else:
                # If the type is unknown, assume it is a box with default size
                dx = dy = dz = 0

            # Handle cases where the position may not have all 3 components
            px = pos[0] if len(pos) > 0 else 0
            py = pos[1] if len(pos) > 1 else 0
            pz = pos[2] if len(pos) > 2 else 0

            min_x = min(min_x, px - dx)
            max_x = max(max_x, px + dx)

            min_y = min(min_y, py - dy)
            max_y = max(max_y, py + dy)

            min_z = min(min_z, pz - dz)
            max_z = max(max_z, pz + dz)

        # Calculate dimensions
        length = max_x - min_x  # Extent along x-axis
        width = max_y - min_y   # Extent along y-axis
        height = max_z - min_z  # Extent along z-axis

        return length, width, height
    
    def local_frame_bounding_box(self, local_center, local_quat, obj_center_pos, obj_half_bounding_box, obj_quat):
        """Calculate the bounding box of an object in a local frame.

        Args:
            local_center (array): position of the local frame's center in the global frame
            local_quat (array): quaternion of the local frame
            obj_center_pos (array): position of the object's center in the global frame
            obj_half_bounding_box (array): half bounding box size of the object
            obj_quat (array): object's quaternion
        Returns:
            bounding_box_local (array): (min_x, max_x, min_y, max_y, min_z, max_z) of the bounding box in the local frame
        """
        # Calculate eight corners of the bounding box in the object's frame
        corners = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1]
        ])
        corners = corners * obj_half_bounding_box
        # Calculate the eight corners in the local frame
        corners_local = np.zeros_like(corners)
        for i, corner in enumerate(corners):
            corners_local[i] = self.local_frame_pos(local_center, local_quat, obj_center_pos, corner, obj_quat)
        # find the min and max of the corners in the local frame
        min_x = np.min(corners_local[:, 0])
        max_x = np.max(corners_local[:, 0])
        min_y = np.min(corners_local[:, 1])
        max_y = np.max(corners_local[:, 1])
        min_z = np.min(corners_local[:, 2])
        max_z = np.max(corners_local[:, 2])

        bounding_box_local = np.array([min_x, max_x, min_y, max_y, min_z, max_z])
        return bounding_box_local
    
    def local_frame_pos(self, local_center, local_quat, obj_center_pos, point_pos, quat):
        """Calculate the position of an point's position in a local frame.

        Args:
            local_center (array): position of the local frame's center in the global frame
            local_quat (array): quaternion of the local frame
            obj_center_pos (array): position of the object's center in the global frame
            point_pos (array): position of the point in the object's frame
            quat (array): object's quaternion
        Returns:
            pos_local (array): position of the point in the local frame
        """
        # Calculate the rotation matrix of the local frame
        local_rot = T.quat2mat(local_quat)

        # Calculate the rotation matrix of the object
        obj_rot = T.quat2mat(quat)

        # Calculate the point's position in the global frame
        pos_global = obj_rot.dot(point_pos) + obj_center_pos

        # Calculate the point's position in the local frame
        pos_local = local_rot.T.dot(pos_global - local_center)

        return pos_local

    def _get_partial_task_metrics(self):
        """
        Returns a dictionary of partial task metrics which correspond to different parts of the task being done.
        """

        # populate with superclass metrics that concern the coffee pod and coffee machine
        metrics = super()._get_partial_task_metrics()

        # whether mug is grasped (NOTE: the use of tolerant grasp function for mug, due to problems with contact)
        metrics["mug_grasp"] = self._check_grasp_tolerant(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.mug.contact_geoms]
        )

        # whether mug has been placed on coffee machine
        metrics["mug_place"] = self.check_mug_placement()

        # new task success includes mug placement
        metrics["task"] = metrics["task"] and metrics["mug_place"]

        # can have a check on drawer being closed here, to make the task even harder
        # print(self.sim.data.qpos[self.cabinet_qpos_addr])

        return metrics


class Coffee_Drawer_Novelty_D0(Coffee_Drawer_Novelty):
    """Rename base class for convenience."""
    pass


class Coffee_Drawer_Novelty_D1(Coffee_Drawer_Novelty_D0):
    """
    Broader initialization for mug (whole right side of table, with rotation) and
    modest movement for coffee machine (some translation and rotation).
    """
    def _get_initial_placement_bounds(self):
        return dict(
            drawer=dict(
                x=(0.15, 0.15),
                y=(-0.35, -0.35),
                z_rot=(np.pi, np.pi),
                reference=self.table_offset,
            ),
            coffee_machine=dict(
                x=(-0.25, -0.15),
                y=(-0.30, -0.25),
                z_rot=(-np.pi / 6., np.pi / 6.),
                reference=self.table_offset,
            ),
            mug=dict(
                x=(-0.15, 0.20),
                y=(0.05, 0.25),
                z_rot=(0.0, 2. * np.pi), 
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.03, 0.03),
                y=(-0.05, 0.03),
                z_rot=(0.0, 0.0),
                reference=np.array((0., 0., 0.)),
            ),
        )

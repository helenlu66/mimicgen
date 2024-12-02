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
            coffee_machine=self.sim.model.body_name2id(self.coffee_machine.root_body),
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

            obj_centric_sensors, obj_centric_sensor_names = self._create_obj_centric_sensors(modality="object_centric")
            sensors += obj_centric_sensors
            names += obj_centric_sensor_names
            actives += [True] * len(obj_centric_sensors)

            # add hinge angle of lid
            @sensor(modality=modality)
            def hinge_angle(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])
            sensors += [hinge_angle]
            names += ["hinge_angle"]
            actives += [True]

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
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            obs_cache[f"{obj_name}_pose"] = obj_pose
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

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
        if hasattr(self, obj_name):
            obj_bounding_box = getattr(self, obj_name).get_bounding_box_half_size()
        elif obj_name == 'coffee_machine_lid':
            obj_bounding_box = self.coffee_machine.lid_size
        elif obj_name == 'coffee_pod_holder':
            obj_bounding_box = self.pod_holder_size
        elif obj_name == 'drawer':
            obj_bounding_box = self.cabinet_object.get_bounding_box_half_size()
        table_z_offset = self.table_offset[2]
        obj_z = self.sim.data.body_xpos[self.obj_body_id[obj_name]][2]
        obj_bottom_z = obj_z - obj_bounding_box[2]
        return obj_bottom_z - table_z_offset < 0.01
    

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
        self.cabinet_object = LongDrawerObject(name="CabinetObject")

        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)
        obj_body = self.cabinet_object
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
        # objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object, self.mug]
        objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object]

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
                x=(0.15, 0.15),
                y=(-0.35, -0.35),
                z_rot=(np.pi, np.pi),
                reference=self.table_offset,
            ),
            coffee_machine=dict(
                x=(-0.15, -0.15),
                y=(-0.25, -0.25),
                z_rot=(-np.pi / 6., -np.pi / 6.),
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
                x=(0.05, 0.20),
                y=(0.05, 0.25),
                z_rot=(0.0, 0.0), 
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.03, 0.03),
                y=(-0.05, 0.03),
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
                mujoco_objects=self.cabinet_object,
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
                ensure_valid_placement=True,
                reference_pos=bounds["mug"]["reference"],
                z_offset=0.02,
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

        self.cabinet_qpos_addr = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])
        self.obj_body_id["drawer"] = self.sim.model.body_name2id(self.cabinet_object.root_body)
        self.obj_body_id["mug"] = self.sim.model.body_name2id(self.mug.root_body)
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id("CabinetObject_drawer_bottom")

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
                if obj is self.cabinet_object:
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
            drawer_rot_mat = T.quat2mat(T.convert_quat(self.sim.model.body_quat[self.sim.model.body_name2id(self.cabinet_object.root_body)], to="xyzw"))
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
        # Positions and orientations of objects
        observables['gripper1_pos'] = observables.pop('robot0_eef_pos')
        observables['gripper1_quat'] = observables.pop('robot0_eef_quat')
        observables['mug1_pos'] = observables.pop('mug_pos')
        observables['mug1_quat'] = observables.pop('mug_quat')
        observables['drawer1_pos'] = observables.pop('drawer_pos')
        observables['drawer1_quat'] = observables.pop('drawer_quat')
        observables['coffee-pod1_pos'] = observables.pop('coffee_pod_pos')
        observables['coffee-pod1_quat'] = observables.pop('coffee_pod_quat')
        observables['coffee-pod-holder1_pos'] = observables.pop('coffee_pod_holder_pos')
        observables['coffee-pod-holder1_quat'] = observables.pop('coffee_pod_holder_quat')
        observables['coffee-machine1_pos'] = observables.pop('coffee_machine_pos')
        observables['coffee-machine1_quat'] = observables.pop('coffee_machine_quat')
        observables['lid1_pos'] = observables.pop('coffee_machine_lid_pos')
        observables['lid1_quat'] = observables.pop('coffee_machine_lid_quat')

        # Distances between pod and pod holder
        observables['coffee-pod1_pos_rel_coffee-pod-holder1'] = observables.pop('pod_pos_rel_pod_holder')
        observables['coffee-pod1_quat_rel_coffee-pod-holder1'] = observables.pop('pod_quat_rel_pod_holder')
        
        # Distances between gripper and objects
        observables['gripper1_to_mug1_dist'] = observables.pop('mug_to_robot0_eef_pos')
        observables['gripper1_to_mug1_quat'] = observables.pop('mug_to_robot0_eef_quat')
        observables['gripper1_to_coffee-pod1-dist'] = observables.pop('coffee_pod_to_robot0_eef_pos')
        observables['gripper1_to_coffee-pod1-quat'] = observables.pop('coffee_pod_to_robot0_eef_quat')
        observables['gripper1_to_coffee-machine1-dist'] = observables.pop('coffee_machine_to_robot0_eef_pos')
        observables['gripper1_to_coffee-machine1-quat'] = observables.pop('coffee_machine_to_robot0_eef_quat')
        observables['gripper1_to_lid1_dist'] = observables.pop('coffee_machine_lid_to_robot0_eef_pos')
        observables['gripper1_to_lid1_quat'] = observables.pop('coffee_machine_lid_to_robot0_eef_quat')
        observables['gripper1_to_drawer1_dist'] = observables.pop('drawer_to_robot0_eef_pos')
        observables['gripper1_to_drawer1_quat'] = observables.pop('drawer_to_robot0_eef_quat')

        # Avoid using drawer joint position as it should not be observable to the robot as a novel object
        # observables.pop('drawer_joint_pos')
        
        # object joint angles
        observables['lid1_hinge_angle'] = observables.pop('hinge_angle')

        return observables
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """

        # super class will populate observables for "mug" and "drawer" poses in addition to coffee machine and coffee pod
        observables = super()._setup_observables()

        drawer_opening_percentage = self.check_drawer_open_percentage()

        if self.use_object_obs:
            modality = "object"

            # add drawer joint angle observable
            # @sensor(modality=modality)
            # def drawer_joint_angle(obs_cache):
            #     return np.array([self.sim.data.qpos[self.cabinet_qpos_addr]])
            # sensors = [drawer_joint_angle]
            # names = ["drawer_joint_angle"]
            # actives = [True]

            # add overlap sensors
            sensors, names = self._create_overlap_sensors()

            @sensor(modality=modality)
            def gripper1_to_any_obj_max_absolute_dist(obs_cache):
                table_size = self.model.mujoco_arena.table_full_size
                # assume the highest the robot can reach is 1.0m above the table
                max_dist = [dist for dist in table_size]  # copy the table size
                max_dist[2] += 1.0
                return max_dist
            sensors += [gripper1_to_any_obj_max_absolute_dist]
            names += ["gripper1_to_any_obj_max_absolute_dist"]

            # add drawer's max travel distance observable
            @sensor(modality=modality)
            def drawer1_cabinet_side_length(obs_cache):
                return self.cabinet_object.drawer_side_wall_size[1]
            sensors += [drawer1_cabinet_side_length]
            names += ["drawer1_cabinet_side_length"]
    

            @sensor(modality=modality)
            def drawer1_travel_distance(obs_cache):
                return drawer_opening_percentage * self.cabinet_object.max_drawer_travel_distance
            sensors += [drawer1_travel_distance]
            names += ["drawer1_travel_distance"]



            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=True,
                )
        
        observables = self.rename_observables(observables)

        return observables
    
    def _calculate_boxes_overlap_percentage(self, obj1_pos, obj1_half_bounding_box, obj2_pos, obj2_half_bounding_box):
        """calculate the percentage of obj1's bounding box that overlaps with obj2's bounding box

        Args:
            obj1_pos (np.array): position of obj1
            obj1_half_bounding_box (np.array): half bounding box size of obj1
            obj2_pos (np.array): position of obj2
            obj2_half_bounding_box (np.array): half bounding box size of obj2
        """
        x1, y1, z1 = obj1_pos
        hx1, hy1, hz1 = obj1_half_bounding_box
        x2, y2, z2 = obj2_pos
        hx2, hy2, hz2 = obj2_half_bounding_box

        # calculate the min and max coordinates of the bounding boxes
        min1 = [x1 - hx1, y1 - hy1, z1 - hz1]
        max1 = [x1 + hx1, y1 + hy1, z1 + hz1]
        min2 = [x2 - hx2, y2 - hy2, z2 - hz2]
        max2 = [x2 + hx2, y2 + hy2, z2 + hz2]

        # calculate the overlap along each axis
        overlap = [max(0, min1[i] - max2[i], max1[i] - min2[i]) for i in range(3)]
        # calculate the volume of the overlap
        overlap_volume = overlap[0] * overlap[1] * overlap[2]
        # calculate the volume of obj1
        obj1_volume = (2 * hx1) * (2 * hy1) * (2 * hz1)
        # calculate the percentage of obj1's volume that overlaps with obj2
        overlap_percentage = overlap_volume / obj1_volume
        return overlap_percentage
    
    def _create_overlap_sensors(self):
        """
        Create sensors for the overlap between objects.
        """
        sensors = []
        names = []

        modality = "object"
        drawer1_id = self.sim.model.body_name2id(self.cabinet_object.root_body)
        drawer1_pos = self.sim.data.body_xpos[drawer1_id]
        drawer1_half_bounding_box = self.cabinet_object.get_bounding_box_half_size()
        drawer1_half_bounding_box[1] = self.cabinet_object.horizontal_radius # use the horizontal radius for the y-axis since for some reason the half bounding box's y is 0
        mug1_id = self.sim.model.body_name2id(self.mug.root_body)
        mug1_pos = self.sim.data.body_xpos[mug1_id]
        mug1_half_bounding_box = self.mug.get_bounding_box_half_size()
        coffee_pod_id = self.sim.model.body_name2id(self.coffee_pod.root_body)
        coffee_pod_pos = self.sim.data.body_xpos[coffee_pod_id]
        coffee_pod_half_bounding_box = self.coffee_pod.get_bounding_box_half_size()
        coffe_machine_lid_id = self.sim.model.body_name2id(self.coffee_machine.name + '_' + self.coffee_machine.lid.root_body)
        coffee_machine_lid_pos = self.sim.data.body_xpos[coffe_machine_lid_id]
        coffee_machine_lid_half_bounding_box = self.coffee_machine.lid.get_bounding_box_half_size()
        coffee_pod_holder_id = self.sim.model.body_name2id(self.coffee_machine.name + '_' + self.coffee_machine.pod_holder.root_body)
        coffee_pod_holder_pos = self.sim.data.body_xpos[coffee_pod_holder_id]
        coffee_pod_holder_half_bounding_box = self.coffee_machine.pod_holder.get_bounding_box_half_size()

        # mug1 overlaps
        mug1_drawer1_overlap = self._calculate_boxes_overlap_percentage(mug1_pos, mug1_half_bounding_box, drawer1_pos, drawer1_half_bounding_box)
        mug1_coffee_pod_holder1_overlap = self._calculate_boxes_overlap_percentage(mug1_pos, mug1_half_bounding_box, coffee_pod_holder_pos, coffee_pod_holder_half_bounding_box)

        # coffee pod overlaps
        coffee_pod1_drawer1_overlap = self._calculate_boxes_overlap_percentage(coffee_pod_pos, coffee_pod_half_bounding_box, drawer1_pos, drawer1_half_bounding_box)
        coffee_pod1_mug1_overlap = self._calculate_boxes_overlap_percentage(coffee_pod_pos, coffee_pod_half_bounding_box, mug1_pos, mug1_half_bounding_box)
        coffee_pod1_coffee_pod_holder1_overlap = self._calculate_boxes_overlap_percentage(coffee_pod_pos, coffee_pod_half_bounding_box, coffee_pod_holder_pos, coffee_pod_holder_half_bounding_box)

        # coffee machine lid overlaps
        coffee_machine_lid1_drawer1_overlap = self._calculate_boxes_overlap_percentage(coffee_machine_lid_pos, coffee_machine_lid_half_bounding_box, drawer1_pos, drawer1_half_bounding_box)
        coffee_machine_lid1_mug1_overlap = self._calculate_boxes_overlap_percentage(coffee_machine_lid_pos, coffee_machine_lid_half_bounding_box, mug1_pos, mug1_half_bounding_box)
        coffee_machine_lid1_coffee_pod_holder1_overlap = self._calculate_boxes_overlap_percentage(coffee_machine_lid_pos, coffee_machine_lid_half_bounding_box, coffee_pod_holder_pos, coffee_pod_holder_half_bounding_box)

        # coffee pod holder overlaps
        coffee_pod_holder1_drawer1_overlap = self._calculate_boxes_overlap_percentage(coffee_pod_holder_pos, coffee_pod_holder_half_bounding_box, drawer1_pos, drawer1_half_bounding_box)
        coffee_pod_holder1_mug1_overlap = self._calculate_boxes_overlap_percentage(coffee_pod_holder_pos, coffee_pod_holder_half_bounding_box, mug1_pos, mug1_half_bounding_box)

        # add sensors
        @sensor(modality=modality)
        def percent_overlap_of_mug1_bounding_box_with_drawer1_bounding_box(obs_cache):
            return np.array([mug1_drawer1_overlap])
        sensors += [percent_overlap_of_mug1_bounding_box_with_drawer1_bounding_box]
        names += ["percent_overlap_of_mug1_bounding_box_with_drawer1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_mug1_bounding_box_with_coffee_pod_holder1_bounding_box(obs_cache):
            return np.array([mug1_coffee_pod_holder1_overlap])
        sensors += [percent_overlap_of_mug1_bounding_box_with_coffee_pod_holder1_bounding_box]
        names += ["percent_overlap_of_mug1_bounding_box_with_coffee-pod-holder1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod1_bounding_box_with_drawer1_bounding_box(obs_cache):
            return np.array([coffee_pod1_drawer1_overlap])
        sensors += [percent_overlap_of_coffee_pod1_bounding_box_with_drawer1_bounding_box]
        names += ["percent_overlap_of_coffee-pod1_bounding_box_with_drawer1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod1_bounding_box_with_mug1_bounding_box(obs_cache):
            return np.array([coffee_pod1_mug1_overlap])
        sensors += [percent_overlap_of_coffee_pod1_bounding_box_with_mug1_bounding_box]
        names += ["percent_overlap_of_coffee-pod1_bounding_box_with_mug1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod1_bounding_box_with_coffee_pod_holder1_bounding_box(obs_cache):
            return np.array([coffee_pod1_coffee_pod_holder1_overlap])
        sensors += [percent_overlap_of_coffee_pod1_bounding_box_with_coffee_pod_holder1_bounding_box]
        names += ["percent_overlap_of_coffee-pod1_bounding_box_with_coffee-pod-holder1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_machine_lid1_bounding_box_with_drawer1_bounding_box(obs_cache):
            return np.array([coffee_machine_lid1_drawer1_overlap])
        sensors += [percent_overlap_of_coffee_machine_lid1_bounding_box_with_drawer1_bounding_box]
        names += ["percent_overlap_of_lid1_bounding_box_with_drawer1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_machine_lid1_bounding_box_with_mug1_bounding_box(obs_cache):
            return np.array([coffee_machine_lid1_mug1_overlap])
        sensors += [percent_overlap_of_coffee_machine_lid1_bounding_box_with_mug1_bounding_box]
        names += ["percent_overlap_of_lid1_bounding_box_with_mug1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_machine_lid1_bounding_box_with_coffee_pod_holder1_bounding_box(obs_cache):
            return np.array([coffee_machine_lid1_coffee_pod_holder1_overlap])
        sensors += [percent_overlap_of_coffee_machine_lid1_bounding_box_with_coffee_pod_holder1_bounding_box]
        names += ["percent_overlap_of_lid1_bounding_box_with_coffee-pod-holder1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod_holder1_bounding_box_with_drawer1_bounding_box(obs_cache):
            return np.array([coffee_pod_holder1_drawer1_overlap])
        sensors += [percent_overlap_of_coffee_pod_holder1_bounding_box_with_drawer1_bounding_box]
        names += ["percent_overlap_of_coffee-pod-holder1_bounding_box_with_drawer1_bounding_box"]

        @sensor(modality=modality)
        def percent_overlap_of_coffee_pod_holder1_bounding_box_with_mug1_bounding_box(obs_cache):
            return np.array([coffee_pod_holder1_mug1_overlap])
        sensors += [percent_overlap_of_coffee_pod_holder1_bounding_box_with_mug1_bounding_box]
        names += ["percent_overlap_of_coffee-pod-holder1_bounding_box_with_mug1_bounding_box"]

        return sensors, names

    def check_in_mug(self, obj_name):
        """
        Returns true if object is in the mug.
        """
        # get mug's half bounding box and pos
        mug_half_bounding_box = self.mug.get_bounding_box_half_size()
        mug_pos = self.sim.data.body_xpos[self.obj_body_id["mug"]]
        # get object's bounding box and pos
        obj_half_bounding_box = getattr(self, obj_name).get_bounding_box_half_size()
        obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
        # check if object is in mug
        in_mug = np.all(np.abs(mug_pos - obj_pos) <= mug_half_bounding_box - obj_half_bounding_box)
        return in_mug
    
    def check_in_drawer(self, obj_name):
        """
        Returns true if object is in the drawer.
        """
        # check if object is in contact with the inside of the drawer
        assert obj_name in ["mug", "coffee_pod"]
        drawer_bottom_geom = "CabinetObject_drawer_bottom"
        obj_in_drawer = self.check_contact(drawer_bottom_geom, getattr(self, obj_name))
        return obj_in_drawer

    
    def check_drawer_open(self):
        """
        Returns true if drawer is open.
        """
        return (self.sim.data.qpos[self.cabinet_qpos_addr] <= -0.13)
    
    def check_drawer_open_percentage(self) -> float:
        """Returns how open the drawer is as a percentage.

        Returns:
            float: Percentage of how open the drawer is (0.0 to 1.0).
        """
        closed_joint_angle = 0.0
        open_joint_angle = -0.13
        joint_angle_range = closed_joint_angle - open_joint_angle # 0.13
        joint_angle = self.sim.data.qpos[self.cabinet_qpos_addr] # might be more negative than -0.13
        if joint_angle < open_joint_angle:
            return 1.0
        elif joint_angle > closed_joint_angle:
            return 0.0
        return (joint_angle - open_joint_angle) / joint_angle_range

    
    def check_mug_upright(self):
        """Returns true if mug is upright. """
        obj_rot = self.sim.data.body_xmat[self.obj_body_id["mug"]].reshape(3, 3)
        z_axis = obj_rot[:3, 2]
        dist_to_z_axis = 1. - z_axis[2]
        return (dist_to_z_axis < 1e-3)
    
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

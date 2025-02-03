# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Simpler object cleanup task (inspired by BUDS Hammer Place, see https://github.com/ARISE-Initiative/robosuite-task-zoo) 
where a single object needs to be packed away into a drawer. The default task is to cleanup a 
particular mug.
"""
import os
import random
from collections import OrderedDict
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import CustomMaterial, add_material, find_elements, string_to_array
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.objects import BallObject, BoxObject


import mimicgen
from mimicgen.models.robosuite.objects import BlenderObject, DrawerObject, LongDrawerObject
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class MugCleanup(SingleArmEnv_MG):
    """
    This class corresponds to the object cleanup task for a single robot arm.

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

            :`'magnitud e'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
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
        camera_names="frontview",
        camera_heights=100,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        shapenet_id="3143a4ac",
        shapenet_scale=0.8,
    ):
        # shapenet mug to use
        self._shapenet_id = shapenet_id
        self._shapenet_scale = shapenet_scale

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

        # # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
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

        # initialize objects of interest
        self._get_drawer_model()
        self._get_object_model()
        # objects = [self.drawer, self.mug]
        objects = [self.drawer, self.cube]

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

    def _get_drawer_model(self):
        """
        Allow subclasses to override which drawer to use - should load into @self.drawer.
        """

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
        self.drawer = DrawerObject(name="DrawerObject")
        obj_body = self.drawer
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                             naming_prefix=obj_body.naming_prefix,
                                                             custom_material=deepcopy(material))
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)

        # Example usage
        XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")
        xml_path = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer.xml")

        # Calculate the bounding length of the cabinet and drawer
        cabinet_length, drawer_length = self.get_cabinet_and_drawer_bounding_length(xml_path)

        # print(f"Cabinet Bounding Length: {cabinet_length}")
        # print(f"Drawer Bounding Length: {drawer_length}")

        # Example usage
        body_name = "drawer_link"  # Replace with the body name you want to measure

        # Get bounding box dimensions
        length, width, height = self.get_bounding_box_dimensions(xml_path, body_name)

        # print(f"Bounding Box Dimensions for '{body_name}':")
        # print(f"Length (X): {length}")
        # print(f"Width (Y): {width}")
        # print(f"Height (Z): {height}")

        # Example usage
        body_name = "base"  # Replace with the body name you want to measure

        # Get bounding box dimensions
        length, width, height = self.get_bounding_box_dimensions(xml_path, body_name)

        # print(f"Bounding Box Dimensions for '{body_name}':")
        # print(f"Length (X): {length}")
        # print(f"Width (Y): {width}")
        # print(f"Height (Z): {height}")


    def _get_object_model(self):
        """
        Allow subclasses to override which object to pack into drawer - should load into @self.mug.
        """
        base_mjcf_path = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/shapenet_core/mugs")
        mjcf_path = os.path.join(base_mjcf_path, "{}/model.xml".format(self._shapenet_id))

        self.mug = BlenderObject(
            name="mug",
            mjcf_path=mjcf_path,
            scale=self._shapenet_scale,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1, 1, 1),
            margin=0.001,
        )

        # Create the cube object
        cube_size = [0.0125, 0.0125, 0.0125]  # Adjust the size to fit inside the mug
        self.cube = BoxObject(
            name="cube",
            size=cube_size,
            rgba=[0, 1, 0, 1],  # Green color
            density=1000,
            friction=(1.0, 0.005, 0.0001),
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
            drawer=dict(
                x=(0., 0.),
                y=(0.3, 0.3),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
            # mug=dict(
            #     x=(0.125, 0.130),
            #      y=(-0.283, -0.278),
            #     z_rot=(5*np.pi/4, 7*np.pi/4), # mug handle points away from the robot
            #     reference=self.table_offset,
            # ),
            # cube=dict(
            #     x=(0.125, 0.130),
            #     y=(-0.283, -0.278),
            #     z_rot=(5*np.pi/4, 7*np.pi/4),
            #     reference=self.table_offset,
            # ),
            mug=dict(
                x=(0.0, 0.1), # mug is placed in table center
                y=(-0.026, -0.016),
                z_rot=(5*np.pi/4, 7*np.pi/4), # mug handle points away from the robot
                reference=self.table_offset,
            ),
            cube=dict(
                x=(0.0, 0.1), # cube is placed in table center inside the mug
                y=(-0.026, -0.016),
                z_rot=(5*np.pi/4, 7*np.pi/4),
                reference=self.table_offset,
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
                name="MugSampler",
                mujoco_objects=self.mug,
                x_range=bounds["mug"]["x"],
                y_range=bounds["mug"]["y"],
                rotation=bounds["mug"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["mug"]["reference"],
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
            mug=self.sim.model.body_name2id(self.mug.root_body),
            drawer=self.sim.model.body_name2id(self.drawer.root_body),
            cube=self.sim.model.body_name2id(self.cube.root_body),
        )
        self.drawer_qpos_addr = self.sim.model.get_joint_qpos_addr(self.drawer.joints[0])
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id("DrawerObject_drawer_bottom")

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
                if obj is self.drawer:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = 0.805 # hardcode z-value to make sure it lies on table surface
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Drawer should start closed (0.) but can set to open (-0.135) for debugging.
        self.sim.data.qpos[self.drawer_qpos_addr] = 0.
        # self.sim.data.qpos[self.drawer_qpos_addr] = -0.135
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

            # add ground-truth poses (absolute and relative to eef) for all objects
            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)

            # add joint position of drawer
            @sensor(modality=modality)
            def drawer_joint_pos(obs_cache):
                return np.array([self.sim.data.qpos[self.drawer_qpos_addr]])
            sensors += [drawer_joint_pos]
            names += ["drawer_joint_pos"]
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
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _check_success(self):
        """
        Check if task is complete.
        """

        # check for closed drawer
        drawer_closed = self.sim.data.qpos[self.drawer_qpos_addr] > -0.01

        # check that object is upright (it shouldn't fall over in the drawer)

        # check z-axis alignment by checking z unit-vector of obj pose and dot with (0, 0, 1)
        # then take cosine dist (1 - dot-prod)
        obj_rot = self.sim.data.body_xmat[self.obj_body_id["mug"]].reshape(3, 3)
        z_axis = obj_rot[:3, 2]
        dist_to_z_axis = 1. - z_axis[2]
        object_upright = (dist_to_z_axis < 1e-3)

        # easy way to check for object in drawer - check if object in contact with bottom drawer geom
        drawer_bottom_geom = "DrawerObject_drawer_bottom"
        object_in_drawer = self.check_contact(drawer_bottom_geom, self.mug)

        return (object_in_drawer and object_upright and drawer_closed)

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

        # Color the gripper visualization site according to its distance to the cleanup object
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.mug)

    def check_directly_on_table(self, obj_name):
        """
        Check if the object is directly on the table.

        Args:
            obj_name (str): Name of the object to check

        Returns:
            bool: True if the object is directly on the table, False otherwise
        """
        if obj_name == "mug":
            return self.check_contact(self.mug, 'table_collision')      
        elif obj_name == "block":
            return self.check_contact(self.cube, 'table_collision') 
        elif obj_name == "drawer":
            return True 
        else:
            raise ValueError("Invalid object name: {}".format(obj_name))

    def check_in_drawer(self, obj_name):
        """
        Check if the object is in the drawer.

        Args:
            obj_name (str): Name of the object to check

        Returns:
            bool: True if the object is in the drawer, False otherwise
        """

        if obj_name == "mug":
            contact = self.check_contact(self.mug, "DrawerObject_drawer_bottom") # Check if the object is in contact with the bottom of the drawer
        elif obj_name == "block":
            obj_name = 'cube'
            contact = self.check_contact(self.cube, "DrawerObject_drawer_bottom") # Check if the object is in contact with the bottom of the drawer
        else:
            raise ValueError("Invalid object name: {}".format(obj_name))
        
        if contact:
            return True
        else: 
            percent_overlap = self.estimate_obj1_overlap_w_obj2(obj_name, "drawer")
            return percent_overlap > 0.5 # obj is considered in the drawer if 50% of it is inside the drawer

            
    def check_in_mug(self, obj_name):
        """
        Check if the object is in the mug.

        Args:
            obj_name (str): Name of the object to check

        Returns:
            bool: True if the object is in the mug, False otherwise
        """
        if obj_name == 'block':
            obj_name = 'cube'
        # get mug's half bounding box and pos
        # mug_half_bounding_box = self.mug.get_bounding_box_half_size()
        # mug_pos = self.sim.data.body_xpos[self.obj_body_id["mug"]]
        # # get object's bounding box and pos
        # obj_half_bounding_box = getattr(self, obj_name).get_bounding_box_half_size()
        # obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
        # # check if object is in mug
        # in_mug = np.all(np.abs(mug_pos - obj_pos) <= mug_half_bounding_box - obj_half_bounding_box)
        # return in_mug
        # optional contact based method that I don't think works as well.
        # return self.check_contact(self.mug, self.cube)
        percent_overlap = self.estimate_obj1_overlap_w_obj2(obj_name, "mug")
        return percent_overlap > 0.5 # obj is considered in the mug if % of it is inside the mug
    
    def check_drawer_open(self):
        """
        Check if the drawer is open.

        Returns:
            bool: True if the drawer is open, False otherwise
        """
        return self.sim.data.qpos[self.drawer_qpos_addr] < -0.01
    
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

    def get_cabinet_and_drawer_bounding_length(self, xml_path: str):
        """
        Computes the maximum bounding box length for the cabinet and drawer.
        
        Args:
        - xml_path (str): Path to the Mujoco XML file generated by MimicGen.
        
        Returns:
        - tuple: A tuple containing (cabinet_length, drawer_length)
        """
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def compute_bounding_length(geoms):
            """
            Computes the maximum bounding box length of a set of geoms.
            Args:
            - geoms (list): List of <geom> elements.
            
            Returns:
            - float: Length of the bounding box.
            """
            min_x, max_x = float('inf'), float('-inf')
            for geom in geoms:
                pos_x = float(geom.get('pos').split()[0])
                size_x = float(geom.get('size').split()[0])
                min_x = min(min_x, pos_x - size_x)
                max_x = max(max_x, pos_x + size_x)
            return max_x - min_x

        # Extract cabinet and drawer geoms
        cabinet_geoms = root.findall(".//body[@name='base']//geom")
        drawer_geoms = root.findall(".//body[@name='drawer_link']//geom")

        # Compute bounding box lengths
        cabinet_length = compute_bounding_length(cabinet_geoms)
        drawer_length = compute_bounding_length(drawer_geoms)

        return cabinet_length, drawer_length
    
    def estimate_obj1_overlap_w_obj2(self, obj1_name, obj2_name):
        """Estimate the percent overlap between two objects' bounding boxes. For example, if obj1 is fully inside obj2, the percentage is 1.

        Args:
            obj1_name (str): name of the first object
            obj2_name (str): name of the second object
        Returns:
            float: percentage of obj1's bounding box that overlaps with obj2's bounding box in [0, 1]
        """
        obj1 = getattr(self, obj1_name)
        obj1_id = self.sim.model.body_name2id(obj1.root_body)
        obj1_pos = self.sim.data.body_xpos[obj1_id]
        obj1_quat = self.sim.data.body_xquat[obj1_id]
        if obj1_name == "drawer":
            XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")
            xml_path = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer.xml")
            body_name = "drawer_link"  
            # Get bounding box dimensions
            length, width, height = self.get_bounding_box_dimensions(xml_path, body_name)
            obj1_half_bounding_box = np.array([length / 2, width / 2, height / 2])
        else:
            obj1_half_bounding_box = obj1.get_bounding_box_half_size()

        obj2 = getattr(self, obj2_name)
        obj2_id = self.sim.model.body_name2id(obj2.root_body)
        obj2_pos = self.sim.data.body_xpos[obj2_id]
        obj2_quat = self.sim.data.body_xquat[obj2_id]
        if obj2_name == "drawer":
            XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")
            xml_path = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer.xml")
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


class MugCleanup_D0(MugCleanup):
    """Rename base class for convenience."""
    pass

# region CubeCleanup_Pre_Novelty
class CubeCleanup_Pre_Novelty(MugCleanup):

    def check_directly_on_table(self, obj_name):
        return super().check_directly_on_table(obj_name)

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
                name="MugSampler",
                mujoco_objects=self.mug,
                x_range=(10,10), # Spawn mug far away
                y_range=(10,10),
                rotation=bounds["mug"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["mug"]["reference"],
                z_offset=0.,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
                x_range=bounds["cube"]["x"],
                y_range=bounds["cube"]["y"],
                rotation=bounds["cube"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["cube"]["reference"],
                z_offset=0.03,
            )
        )

# region CubeCleanup_Mug_Novelty
class CubeCleanup_Mug_Novelty(MugCleanup):

    def _reset_internal(self):
        super()._reset_internal()
        self._place_cube_in_mug()

    def _place_cube_in_mug(self):
        """
        Place the cube inside the mug after the mug's position has been set.
        """
        # Get the mug's position and orientation
        mug_body_id = self.sim.model.body_name2id(self.mug.root_body)
        mug_pos = self.sim.data.body_xpos[mug_body_id]
        mug_quat = T.convert_quat(self.sim.data.body_xquat[mug_body_id], to='xyzw')

        # Calculate the cube's position relative to the mug
        # Assuming the mug's opening is along the positive z-axis
        cube_offset = np.array([0.0, -0.01, -0.01])  # Adjust the offset to place the cube inside the mug

        # Rotate the offset by the mug's orientation
        cube_offset_rotated = T.quat2mat(mug_quat).dot(cube_offset)

        # Set the cube's position
        cube_pos = mug_pos + cube_offset_rotated
        cube_quat = mug_quat  # Align the cube's orientation with the mug's orientation

        # Set the cube's joint position
        self.sim.data.set_joint_qpos(
            self.cube.joints[0], np.concatenate([cube_pos, cube_quat])
        )
        self.sim.forward()
    
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
        observables['block1_pos'] = observables.pop('cube_pos')
        observables['block1_quat'] = observables.pop('cube_quat')
        
        # Distances between gripper and objects
        observables['gripper1_to_mug1_dist'] = observables.pop('mug_to_robot0_eef_pos')
        observables['gripper1_to_mug1_quat'] = observables.pop('mug_to_robot0_eef_quat')
        observables['gripper1_to_block1_dist'] = observables.pop('cube_to_robot0_eef_pos')
        observables['gripper1_to_block1_quat'] = observables.pop('cube_to_robot0_eef_quat')
        observables['gripper1_to_drawer1_dist'] = observables.pop('drawer_to_robot0_eef_pos')
        observables['gripper1_to_drawer1_quat'] = observables.pop('drawer_to_robot0_eef_quat')

        observables['drawer1_joint_pos'] = observables.pop('drawer_joint_pos')

        return observables

    
    def _create_overlap_sensors(self):
        """
        Create sensors for the overlap between objects.
        """
        sensors = []
        names = []

        # Get robot prefix and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"
        block1_mug1_overlap = self.estimate_obj1_overlap_w_obj2("cube", "mug")
        block1_drawer1_overlap = self.estimate_obj1_overlap_w_obj2("cube", "drawer")
        mug1_drawer1_overlap = self.estimate_obj1_overlap_w_obj2("mug", "drawer")
        mug1_block1_overlap = self.estimate_obj1_overlap_w_obj2("mug", "cube")
        drawer1_block1_overlap = self.estimate_obj1_overlap_w_obj2("drawer", "cube")
        drawer1_mug1_overlap = self.estimate_obj1_overlap_w_obj2("drawer", "mug")

        @sensor(modality=modality)
        def percent_overlap_of_block1_with_mug1(obs_cache):
            return block1_mug1_overlap
        sensors += [percent_overlap_of_block1_with_mug1]
        names += ["percent_overlap_of_block1_with_mug1"]

        @sensor(modality=modality)
        def percent_overlap_of_block1_with_drawer1(obs_cache):
            return block1_drawer1_overlap
        sensors += [percent_overlap_of_block1_with_drawer1]
        names += ["percent_overlap_of_block1_with_drawer1"]

        @sensor(modality=modality)
        def percent_overlap_of_mug1_with_drawer1(obs_cache):
            return mug1_drawer1_overlap
        sensors += [percent_overlap_of_mug1_with_drawer1]
        names += ["percent_overlap_of_mug1_with_drawer1"]

        @sensor(modality=modality)
        def percent_overlap_of_mug1_with_block1(obs_cache):
            return mug1_block1_overlap
        sensors += [percent_overlap_of_mug1_with_block1]
        names += ["percent_overlap_of_mug1_with_block1"]

        @sensor(modality=modality)
        def percent_overlap_of_drawer1_with_block1(obs_cache):
            return drawer1_block1_overlap
        sensors += [percent_overlap_of_drawer1_with_block1]
        names += ["percent_overlap_of_drawer1_with_block1"]

        @sensor(modality=modality)
        def percent_overlap_of_drawer1_with_mug1(obs_cache):
            return drawer1_mug1_overlap
        sensors += [percent_overlap_of_drawer1_with_mug1]
        names += ["percent_overlap_of_drawer1_with_mug1"]

        return sensors, names

    def _setup_observables(self):
        observables = super()._setup_observables()

        observables = self.rename_observables(observables)

        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            block1_id = self.sim.model.body_name2id(self.cube.root_body)
            drawer1_id = self.sim.model.body_name2id(self.drawer.root_body)
            mug1_id = self.sim.model.body_name2id(self.mug.root_body)

            # Add cube sensors
            block_sensors, block_sensor_names = self._create_obj_sensors(
                obj_name="cube", modality=modality
            )

            # Add overlap sensors
            sensors, names = self._create_overlap_sensors()

            sensors += block_sensors
            names += block_sensor_names

            @sensor(modality=modality)
            def gripper1_to_obj_max_possible_dist(obs_cache):
                table_size = self.model.mujoco_arena.table_full_size
                # assume the highest the robot can reach is 0.9m above the table
                max_dist = [dist for dist in table_size]  # copy the table size
                max_dist[2] += 0.9
                return max_dist
            sensors += [gripper1_to_obj_max_possible_dist]
            names += ["gripper1_to_obj_max_possible_dist"]

            @sensor(modality=modality)
            def obj_max_possible_height_above_table1_surface(obs_cache):
                return 0.9 # assume the highest an object can be is 0.9m above the table
            sensors += [obj_max_possible_height_above_table1_surface]
            names += ["obj_max_possible_height_above_table1_surface"]

            @sensor(modality=modality)
            def height_of_block1_lowest_point_above_table1_surface(obs_cache):
                table1_height = self.table_offset[2]
                block1_pos = self.sim.data.body_xpos[block1_id]
                # estimate the lowest point to be half bounding box below the center
                block1_half_bounding_box = self.cube.get_bounding_box_half_size()
                lowest_block1_point = block1_pos[2] - block1_half_bounding_box[2]
                return max(0, lowest_block1_point - table1_height) # make sure it's non-negative
            sensors += [height_of_block1_lowest_point_above_table1_surface]
            names += ["height_of_block1_lowest_point_above_table1_surface"]

            @sensor(modality=modality)
            def height_of_mug1_lowest_point_above_table1_surface(obs_cache):
                table1_height = self.table_offset[2]
                mug1_pos = self.sim.data.body_xpos[mug1_id]
                # estimate the lowest point to be half bounding box below the center
                mug1_half_bounding_box = self.mug.get_bounding_box_half_size()
                lowest_mug1_point = mug1_pos[2] - mug1_half_bounding_box[2]
                return max(0, lowest_mug1_point - table1_height) # make sure it's non-negative
            sensors += [height_of_mug1_lowest_point_above_table1_surface]
            names += ["height_of_mug1_lowest_point_above_table1_surface"]

            @sensor(modality=modality)
            def height_of_drawer1_lowest_point_above_table1_surface(obs_cache):
                table1_height = self.table_offset[2]
                drawer1_pos = self.sim.data.body_xpos[drawer1_id]
                # estimate the lowest to be half bounding box below the center
                drawer1_half_bounding_box = self.drawer.get_bounding_box_half_size()
                lowest_drawer1_point = drawer1_pos[2] - drawer1_half_bounding_box[2]
                return max(0, lowest_drawer1_point - table1_height) # make sure it's non-negative
            sensors += [height_of_drawer1_lowest_point_above_table1_surface]
            names += ["height_of_drawer1_lowest_point_above_table1_surface"]


            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=True,
                )
        return observables

        

# region MugCleanup_D1
class MugCleanup_D1(MugCleanup_D0):
    """
    Wider initialization for both drawer and object.
    """
    def _get_initial_placement_bounds(self):
        return dict(
            drawer=dict(
                x=(-0.15, 0.05),
                y=(0.25, 0.35),
                z_rot=(-np.pi / 6., np.pi / 6.),
                reference=self.table_offset,
            ),
            object=dict(
                x=(-0.25, 0.15),
                y=(-0.3, -0.15),
                z_rot=(0., 2. * np.pi),
                reference=self.table_offset,
            ),
        )
    
    def check_direcctly_on_table(self, obj_name):
        return super().check_direcctly_on_table(obj_name)

# region MugCleanup_O1
class MugCleanup_O1(MugCleanup_D0):
    """
    Use different mug.
    """
    def __init__(
        self,
        **kwargs,
    ):
        super(MugCleanup_O1, self).__init__(
            shapenet_id="34ae0b61",
            shapenet_scale=0.8,
            **kwargs,
        )

# region MugCleanup_O2
class MugCleanup_O2(MugCleanup_D0):
    """
    Use a random mug on each episode reset.
    """
    def __init__(
        self,
        **kwargs,
    ):
        # list of tuples - (shapenet_id, shapenet_scale)
        self._assets = [
            ("3143a4ac", 0.8),          # beige round mug
            ("34ae0b61", 0.8),          # bronze mug with green inside
            ("128ecbc1", 0.66666667),   # light blue round mug, thicker boundaries
            ("d75af64a", 0.66666667),   # off-white cylindrical tapered mug
            ("5fe74bab", 0.8),          # brown mug, thin boundaries
            ("345d3e72", 0.66666667),   # black round mug
            ("48e260a6", 0.66666667),   # red round mug 
            ("8012f52d", 0.8),          # yellow round mug with bigger base 
            ("b4ae56d6", 0.8),          # yellow cylindrical mug 
            ("c2eacc52", 0.8),          # wooden cylindrical mug
            ("e94e46bc", 0.8),          # dark blue cylindrical mug
            ("fad118b3", 0.66666667),   # tall green cylindrical mug
        ]
        self._base_mjcf_path = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/shapenet_core/mugs")
        super(MugCleanup_O2, self).__init__(shapenet_id=None, shapenet_scale=None, **kwargs)

    def _get_object_model(self):
        """
        Allow subclasses to override which object to pack into drawer - should load into @self.mug.
        """
        self._shapenet_id, self._shapenet_scale = random.choice(self._assets)
        mjcf_path = os.path.join(self._base_mjcf_path, "{}/model.xml".format(self._shapenet_id))

        self.mug = BlenderObject(
            name="mug",
            mjcf_path=mjcf_path,
            scale=self._shapenet_scale,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1, 1, 1),
            margin=0.001,
        )

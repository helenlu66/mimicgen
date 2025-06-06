# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import mujoco
import numpy as np
from six import with_metaclass

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.nut_assembly import NutAssembly, NutAssemblySquare
from robosuite.models.arenas import PegsArena
from robosuite.models.objects import SquareNutObject, RoundNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, find_elements
from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_distance


from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG

#region Novelty Environment 
class NutAssembly_D0_RoundPeg_Novelty(NutAssembly, SingleArmEnv_MG):
    """Class NutAssembly_D0_RoundPeg_Novelty
    Augments the robosuite nut assembly task for mimicgen.
    Methods:
        __init__(**kwargs):
            Initializes the NutAssembly_D0_RoundPeg_Novelty class with a custom placement initializer.
        edit_model_xml(xml_str):
            Edits the model XML string to avoid conflicts in function implementation.
        _get_initial_placement_bounds():
            Internal function to get bounds for randomization of initial placements of objects.
            Returns a dictionary with the structure:
                    reference: np array of shape (3,) for reference position in world frame
        check_directly_on_table(obj_name):
            Checks if the specified object is directly on the table.
                obj_name (str): The object name
        check_on_peg(nut_name, peg_name):
            Checks if the specified nut is on the specified peg.
                nut_name (str): The nut name
                peg_name (str): The peg name
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        # make placement initializer here
        nut_names = ("SquareNut", "RoundNut")

        bounds = self._get_initial_placement_bounds()
        nut_x_ranges = (bounds["square_nut"]["x"], bounds["round_nut"]["x"])
        nut_y_ranges = (bounds["square_nut"]["y"], bounds["round_nut"]["y"])
        nut_z_ranges = (bounds["square_nut"]["z_rot"], bounds["round_nut"]["z_rot"])
        nut_references = (bounds["square_nut"]["reference"], bounds["round_nut"]["reference"])
        z_offsets = (-0.02, -0.04)

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for nut_name, x_range, y_range, z_range, z_offset, ref in zip(nut_names, nut_x_ranges, nut_y_ranges, nut_z_ranges, z_offsets, nut_references):
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{nut_name}Sampler",
                    x_range=x_range,
                    y_range=y_range,
                    rotation=z_range,
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=ref,
                    z_offset=z_offset,
                )
            )

        NutAssembly.__init__(self, placement_initializer=placement_initializer, **kwargs)

        # self.rename_observables()

    

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

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
        square_peg_x = -0.05
        square_peg_y = 0.1
        round_peg_x = -0.05
        round_peg_y = -0.1
        return dict(
            square_nut=dict(
                x=(round_peg_x, round_peg_x),
                y=(round_peg_y, round_peg_y),
                z_rot=(47*np.pi/48, 49*np.pi/48),
                reference=np.array((0, 0, 0.82)),
            ),
            round_nut=dict(
                x=(round_peg_x, round_peg_x),
                y=(round_peg_y, round_peg_y),
                z_rot=(0., 0), # to make the round nut point away from the robot so it's not in the way
                reference=np.array((0, 0, 0.82)),
            ),
            square_peg=dict(
                x=(square_peg_x, square_peg_x),
                y=(square_peg_y, square_peg_y),
                z_rot=(0., 0.),
                reference=np.array((square_peg_x, square_peg_y, 0.83)),
            ),
            round_peg=dict(
                x=(round_peg_x, round_peg_x),
                y=(round_peg_y, round_peg_y),
                z_rot=(0., 0.),
                reference=np.array((round_peg_x, round_peg_y, 0.85)),
            ),
        )

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena
    

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation 
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # Change the positions of the pegs
        square_peg_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")  # get xml entry
        square_peg_bounds = self._get_initial_placement_bounds()["square_peg"]  # get placement bounds
        square_peg_xml_pos = string_to_array(square_peg_xml.get("pos"))  # get position
        square_peg_xml_pos[0] = square_peg_bounds["reference"][0]  # update x position
        square_peg_xml.set("pos", array_to_string(square_peg_xml_pos))  # set new position
        
        round_peg_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")
        round_peg_bounds = self._get_initial_placement_bounds()["round_peg"]
        round_peg_xml_pos = string_to_array(round_peg_xml.get("pos"))
        round_peg_xml_pos[0] = round_peg_bounds["reference"][0]
        round_peg_xml.set("pos", array_to_string(round_peg_xml_pos))        

        # additional code for applying randomization - do before setting the new position
        # sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        # sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        # sample_z_rot = np.random.uniform(low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1])
        # square_peg_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        # square_peg_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        # square_peg_xml_quat = np.array([np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)])

        # # set modified entry in xml
        # square_peg_xml.set("pos", array_to_string(square_peg_xml_pos))
        # square_peg_xml.set("quat", array_to_string(square_peg_xml_quat))
        # round_peg_xml.set("pos", array_to_string(round_peg_xml_pos))

        # get collision checking entries
        self.square_peg_size = string_to_array(square_peg_xml.find("./geom").get("size"))
        self.round_peg_size = string_to_array(round_peg_xml.find("./geom").get("size"))
        self.square_peg_horizontal_radius = np.linalg.norm(self.square_peg_size[0:2], 2)
        self.round_peg_horizontal_radius = np.linalg.norm(self.round_peg_size[0:2], 2)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.nuts,
        )
    
    #region Nov Observers
    def rename_observables(self, observables):
        """
        Make copies of the observables with names that match the planning domain.
        """
        # Relative positions of objects
        observables['square-nut1_to_gripper1_pos'] = observables.pop('SquareNut_to_robot0_eef_pos')
        observables['round-nut1_to_gripper1_pos'] = observables.pop('RoundNut_to_robot0_eef_pos')
        observables['square-nut1_handle_to_gripper1_pos'] = observables.pop('SquareNut_handle_to_robot0_eef_pos')
        observables['round-nut1_handle_to_gripper1_pos'] = observables.pop('RoundNut_handle_to_robot0_eef_pos')
        
        # Positions of gripper
        observables['gripper1_pos'] = observables.pop('robot0_eef_pos')
        observables['gripper1_aperture'] = observables.pop('robot0_gripper_aperture')
        observables['griper1_fingers_qpos'] = observables.pop('robot0_gripper_fingers_qpos')

        # Positions of objects
        # observables['gripper1_pos'] = observables.pop('robot0_eef_pos')
        # observables['griper1_fingers_qpos'] = observables.pop('robot0_gripper_fingers_qpos')
        # observables['square-nut1_pos'] = observables.pop('SquareNut_pos')
        # observables['square-nut1_handle_pos'] = observables.pop('SquareNut_handle_pos')
        # observables['round-nut1_pos'] = observables.pop('RoundNut_pos')
        # observables['round-nut1_handle_pos'] = observables.pop('RoundNut_handle_pos')
        
        # Orientations of objects
        # observables['gripper1_euler_angles'] = observables.pop('robot0_eef_euler_angles')
        # observables['square-nut1_euler_angles'] = observables.pop('SquareNut_euler_angles')
        # observables['square-nut1_handle_euler_angles'] = observables.pop('SquareNut_handle_euler_angles')
        # observables['round-nut1_euler_angles'] = observables.pop('RoundNut_euler_angles')
        # observables['round-nut1_handle_euler_angles'] = observables.pop('RoundNut_handle_euler_angles')

        return observables


    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        observables = self.rename_observables(observables)

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            square_nut_id = self.sim.model.body_name2id("SquareNut_main")
            round_nut_id = self.sim.model.body_name2id("RoundNut_main")
            square_peg_id = self.sim.model.body_name2id("peg1")
            round_peg_id = self.sim.model.body_name2id("peg2")
            gripper_id = self.sim.model.body_name2id("gripper0_eef")

            square_peg_pos = np.copy(self.sim.data.body_xpos[square_peg_id])
            top_of_square_peg = square_peg_pos
            top_of_square_peg[2] = self.square_peg_size[2] + square_peg_pos[2] 

            round_peg_pos = np.copy(self.sim.data.body_xpos[round_peg_id])
            top_of_round_peg = round_peg_pos
            top_of_round_peg[2] = self.round_peg_size[1] + round_peg_pos[2]  # index 1 is the height because index 0 is radius

            # get collision distances
            # robot_peg_collision_dist, robot_peg_closest_point = self.sim.robot_obj_collision_dist('mount0')
            # robot_table_collision_dist, robot_table_closest_point = self.sim.robot_obj_collision_dist('table')
            # robot_square_nut_collision_dist, robot_square_nut_closest_point = self.sim.robot_obj_collision_dist('SquareNut')
            # robot_round_nut_collision_dist, robot_round_nut_closest_point = self.sim.robot_obj_collision_dist('RoundNut')

            # add sensors for collision distance and closest points
            # @sensor(modality=modality)
            # def robot_body_to_peg_closest_point(obs_cache):
            #     return robot_peg_closest_point
            # sensors = [robot_body_to_peg_closest_point]
            # names = ["robot_body_to_square-peg1_closest_point"]
            # actives = [True]

            # @sensor(modality=modality)
            # def robot_body_to_table_closest_point(obs_cache):
            #     return robot_table_closest_point
            # sensors += [robot_body_to_table_closest_point]
            # names += ["robot_body_to_table1_closest_point"]
            # actives += [True]

            # @sensor(modality=modality)
            # def robot_body_to_square_nut_closest_point(obs_cache):
            #     return robot_square_nut_closest_point
            # sensors += [robot_body_to_square_nut_closest_point]
            # names += ["robot_body_to_square-nut1_closest_point"]
            # actives += [True]

            # @sensor(modality=modality)
            # def robot_body_to_round_nut_closest_point(obs_cache):
            #     return robot_round_nut_closest_point
            # sensors += [robot_body_to_round_nut_closest_point]
            # names += ["robot_body_to_round-nut1_closest_point"]
            # actives += [True]

            @sensor(modality='object_collision')
            def robot_body_to_peg_collision_dist(obs_cache):
                smallest_dist = np.inf
                closest_point = [0, 0, 0]
                assert hasattr(self.sim.model, "geom_dists"), "geom_dists not computed."
                for robot_geom in self.sim.model.geom_dists:
                    for obj_geom_name in self.sim.model.geom_dists[robot_geom]:
                        if 'peg1' not in obj_geom_name:
                            continue
                        else:
                            smallest_dist = self.sim.model.geom_dists[robot_geom][obj_geom_name]['dist']
                            closest_point = self.sim.model.geom_dists[robot_geom][obj_geom_name]['closest_point']
                return [smallest_dist] + list(closest_point)
            sensors = [robot_body_to_peg_collision_dist]
            names = ["robot_body_to_square-peg1_collision_dist"]
            actives = [False]

            @sensor(modality='object_collision')
            def robot_body_to_table_collision_dist(obs_cache):
                smallest_dist = np.inf
                closest_point = [None, None, None]
                assert hasattr(self.sim.model, "geom_dists"), "geom_dists not computed."
                for robot_geom in self.sim.model.geom_dists:
                    for obj_geom_name in self.sim.model.geom_dists[robot_geom]:
                        if 'table' not in obj_geom_name:
                            continue
                        else:
                            smallest_dist = self.sim.model.geom_dists[robot_geom][obj_geom_name]['dist']
                            closest_point = self.sim.model.geom_dists[robot_geom][obj_geom_name]['closest_point']
                return [smallest_dist] + list(closest_point)
            sensors += [robot_body_to_table_collision_dist]
            names += ["robot_body_to_table1_collision_dist"]
            actives += [False]

            @sensor(modality='object_collision')
            def robot_body_to_square_nut_collision_dist(obs_cache):
                smallest_dist = np.inf
                closest_point = [None, None, None]
                assert hasattr(self.sim.model, "geom_dists"), "geom_dists not computed."
                for robot_geom in self.sim.model.geom_dists:
                    for obj_geom_name in self.sim.model.geom_dists[robot_geom]:
                        if 'SquareNut' not in obj_geom_name:
                            continue
                        else:
                            smallest_dist = self.sim.model.geom_dists[robot_geom][obj_geom_name]['dist']
                            closest_point = self.sim.model.geom_dists[robot_geom][obj_geom_name]['closest_point']
                return [smallest_dist] + list(closest_point)
            sensors += [robot_body_to_square_nut_collision_dist]
            names += ["robot_body_to_square-nut1_collision_dist"]
            actives += [False]

            @sensor(modality='object_collision')
            def robot_body_to_round_nut_collision_dist(obs_cache):
                smallest_dist = np.inf
                closest_point = [None, None, None]
                assert hasattr(self.sim.model, "geom_dists"), "geom_dists not computed."
                for robot_geom in self.sim.model.geom_dists:
                    for obj_geom_name in self.sim.model.geom_dists[robot_geom]:
                        if 'RoundNut' not in obj_geom_name:
                            continue
                        else:
                            smallest_dist = self.sim.model.geom_dists[robot_geom][obj_geom_name]['dist']
                            closest_point = self.sim.model.geom_dists[robot_geom][obj_geom_name]['closest_point']
                return [smallest_dist] + list(closest_point)

            
            @sensor(modality=modality)
            def square_peg1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[square_peg_id])

            @sensor(modality=modality)
            def square_peg1_to_gripper1_pos(obs_cache):
                # Immediately return default value if cache is empty
                if "world_pose_in_gripper" not in obs_cache:
                    return np.zeros(3)
                square_peg_pos = self.sim.data.body_xpos[square_peg_id]
                square_peg_quat = self.sim.data.body_xquat[square_peg_id]
                square_peg_pose = T.pose2mat((square_peg_pos, square_peg_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(square_peg_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"square_peg1_to_gripper1_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def square_peg1_euler_angles(obs_cache):
                return T.mat2euler(T.quat2mat(T.convert_quat(self.sim.data.body_xquat[square_peg_id], to="xyzw")))

            @sensor(modality=modality)
            def round_peg1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[round_peg_id])
            
            @sensor(modality=modality)
            def round_peg1_to_gripper1_pos(obs_cache):
                # Immediately return default value if cache is empty
                if "world_pose_in_gripper" not in obs_cache:
                    return np.zeros(3)
                round_peg_pos = self.sim.data.body_xpos[round_peg_id]
                round_peg_quat = self.sim.data.body_xquat[round_peg_id]
                round_peg_pose = T.pose2mat((round_peg_pos, round_peg_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(round_peg_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"round_peg1_to_gripper1_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def round_peg1_euler_angles(obs_cache):
                return T.mat2euler(T.quat2mat(T.convert_quat(self.sim.data.body_xquat[round_peg_id], to="xyzw")))
            

            @sensor(modality=modality)
            def gripper1_to_obj_max_possible_dist(obs_cache):
                table_size = self.model.mujoco_arena.table_full_size
                # assume the highest the robot can reach is 0.9m above the table
                max_dist = [dist for dist in table_size]  # copy the table size
                max_dist[2] += 0.9
                return max_dist

            @sensor(modality=modality)
            def gripper1_to_square_peg1_dist(obs_cache):
                gripper_pos = self.sim.data.body_xpos[gripper_id]
                peg_pos = self.sim.data.body_xpos[square_peg_id]
                return gripper_pos - peg_pos

            @sensor(modality=modality)
            def gripper1_to_square_peg1_quat(obs_cache):
                gripper_quat = self.sim.data.body_xquat[gripper_id]
                peg_quat = self.sim.data.body_xquat[square_peg_id]
                return quat_distance(gripper_quat, peg_quat)

            @sensor(modality=modality)
            def gripper1_to_round_peg1_dist(obs_cache):
                gripper_pos = self.sim.data.body_xpos[gripper_id]
                peg_pos = self.sim.data.body_xpos[round_peg_id]
                return gripper_pos - peg_pos

            @sensor(modality=modality)
            def gripper1_to_round_peg1_quat(obs_cache):
                gripper_quat = self.sim.data.body_xquat[gripper_id]
                peg_quat = self.sim.data.body_xquat[round_peg_id]
                return quat_distance(gripper_quat, peg_quat)

            @sensor(modality=modality)
            def square_peg1_height(obs_cache):
                return top_of_square_peg[2] - 0.82

            @sensor(modality=modality)
            def round_peg1_height(obs_cache):
                return top_of_round_peg[2] - 0.82

            @sensor(modality=modality)
            def square_nut1_bottom_height_above_square_peg1_base(obs_cache):
                bottom_of_square_nut = np.copy(self.sim.data.body_xpos[square_nut_id]) - [0, 0, 0.01]
                bottom_of_square_peg1 = 0.82
                return bottom_of_square_nut[2] - bottom_of_square_peg1

            @sensor(modality=modality)
            def square_nut1_bottom_height_above_round_peg1_base(obs_cache):
                bottom_of_square_nut = np.copy(self.sim.data.body_xpos[square_nut_id]) - [0, 0, 0.01]
                bottom_of_round_peg1 = 0.82
                return bottom_of_square_nut[2] - bottom_of_round_peg1

            @sensor(modality=modality)
            def round_nut1_bottom_height_above_square_peg1_base(obs_cache):
                bottom_of_round_nut = np.copy(self.sim.data.body_xpos[round_nut_id]) - [0, 0, 0.01]
                bottom_of_square_peg1 = 0.82
                return bottom_of_round_nut[2] - bottom_of_square_peg1

            @sensor(modality=modality)
            def round_nut1_bottom_height_above_round_peg1_base(obs_cache):
                bottom_of_round_nut = np.copy(self.sim.data.body_xpos[round_nut_id]) - [0, 0, 0.01]
                bottom_of_round_peg1 = 0.82
                return bottom_of_round_nut[2] - bottom_of_round_peg1

            # include only the two relative pos
            names += ["square_peg1_to_gripper1_pos", "round_peg1_to_gripper1_pos"]
            sensors += [square_peg1_to_gripper1_pos, round_peg1_to_gripper1_pos]
            actives += [True, True]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables
    

    def robot_obj_collision_dist(self, obj_name):
        """Get the smallest distance between the robots in the scene and the object.

        Args:
            obj_name (str): name of the object
        """
        smallest_dist = np.inf
        assert hasattr(self, "geom_dists"), "geom_dists not computed."
        for robot_geom in self.geom_dists:
            for obj_geom_name in self.geom_dists[robot_geom]:
                if obj_name in obj_geom_name:
                    smallest_dist = min(smallest_dist, self.geom_dists[robot_geom][obj_geom_name])
        return smallest_dist
    
    #region Novelty Detectors
    def check_directly_on_table(self, obj_name):
        """
        Check if the object is directly on the table.

        Args:
            obj_name (str): the object name

        Returns:
            bool: True if the object is directly on the table
        """
        if obj_name == "round-nut":
            round_nut_geoms = ["RoundNut_g0", "RoundNut_g1", "RoundNut_g2", "RoundNut_g3", 
                "RoundNut_g4", "RoundNut_g5", "RoundNut_g6", "RoundNut_g7", "RoundNut_g8"]
            return any(self.check_contact(geom, 'table_collision') for geom in round_nut_geoms)     
        elif obj_name == "square-nut":
            square_nut_geoms = ["SquareNut_g0", "SquareNut_g1", "SquareNut_g2", "SquareNut_g3", 
                "SquareNut_g4"]
            return any(self.check_contact(geom, 'table_collision') for geom in square_nut_geoms)
        elif obj_name == "round-peg" or obj_name == "square-peg":
            return True  # Pegs are always on the table
        else:
            raise ValueError("Invalid object name: {}".format(obj_name))

    def check_on_peg(self, nut_name, peg_name):
        '''
        Check if the nut is on the peg
        
        Args:
            nut_name (str): The nut
            peg_name (str): The peg

        Returns:
            bool: True if the nut is on the peg
        '''
        if nut_name == "round-nut":
            nut_name = "RoundNut"
        elif nut_name == "square-nut":
            nut_name = "SquareNut"

        if peg_name == "round-peg":
            peg_id = 1
        elif peg_name == "square-peg":
            peg_id = 0
        
        nut_pos = self.sim.data.body_xpos[self.obj_body_id[nut_name]]
        return self.on_peg(nut_pos, peg_id)

#region Pre-Novelty Environment    
class NutAssembly_D0_Pre_Novelty(NutAssembly_D0_RoundPeg_Novelty):
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
            square_nut=dict(
                x=(-0.115, -0.11),
                y=(0.11, 0.225),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            round_nut=dict(
                x=(-0.115, -0.11),
                y=(-0.225, -0.11),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg=dict(
                x=(-0.1, 0.3),
                y=(-0.2, 0.2),
                z_rot=(0., 0.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation 
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        square_peg_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        round_peg_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        square_peg_xml_pos = string_to_array(square_peg_xml.get("pos"))
        peg_bounds = self._get_initial_placement_bounds()["peg"]

        sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        sample_z_rot = np.random.uniform(low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1])
        square_peg_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        square_peg_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        square_peg_xml_quat = np.array([np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)])

        # move peg2 completely out of scene
        round_peg_xml_pos = string_to_array(square_peg_xml.get("pos"))
        round_peg_xml_pos[0] = -10.
        round_peg_xml_pos[1] = 0.

        # set modified entry in xml
        square_peg_xml.set("pos", array_to_string(square_peg_xml_pos))
        square_peg_xml.set("quat", array_to_string(square_peg_xml_quat))
        round_peg_xml.set("pos", array_to_string(round_peg_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(square_peg_xml.find("./geom").get("size"))
        peg2_size = string_to_array(round_peg_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.nuts,
        )

    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            peg1_id = self.sim.model.body_name2id("peg1")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[peg1_id])

            name = "peg1_pos"
            observables[name] = Observable(
                name=name,
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )

        return observables

#region Default Environments
class Square_D0(NutAssemblySquare, SingleArmEnv_MG):
    """
    Augment robosuite nut assembly square task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        # make placement initializer here
        nut_names = ("SquareNut", "RoundNut")

        # note: makes round nut init somewhere far off the table
        round_nut_far_init = (-1.1, -1.0)

        bounds = self._get_initial_placement_bounds()
        nut_x_ranges = (bounds["nut"]["x"], bounds["nut"]["x"])
        nut_y_ranges = (bounds["nut"]["y"], round_nut_far_init)
        nut_z_ranges = (bounds["nut"]["z_rot"], bounds["nut"]["z_rot"])
        nut_references = (bounds["nut"]["reference"], bounds["nut"]["reference"])

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for nut_name, x_range, y_range, z_range, ref in zip(nut_names, nut_x_ranges, nut_y_ranges, nut_z_ranges, nut_references):
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{nut_name}Sampler",
                    x_range=x_range,
                    y_range=y_range,
                    rotation=z_range,
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=ref,
                    z_offset=0.02,
                )
            )

        NutAssemblySquare.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

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
            nut=dict(
                x=(0.23, 0.23),
                y=(0.1, 0.1),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )


class Square_D1(Square_D0):
    """
    Specifies a different placement initializer for the pegs where it is initialized
    with a broader x-range and broader y-range.
    """
    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0.115, 0.115),
                y=(-0.255, 0.255),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg=dict(
                x=(-0.1, 0.3),
                y=(-0.2, 0.2),
                z_rot=(0., 0.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation 
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        square_peg_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        round_peg_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        square_peg_xml_pos = string_to_array(square_peg_xml.get("pos"))
        round_peg_xml_pos = string_to_array(round_peg_xml.get("pos"))

        # set modified entry in xml
        square_peg_xml.set("pos", array_to_string(square_peg_xml_pos))
        square_peg_xml.set("quat", array_to_string(square_peg_xml_quat))
        round_peg_xml.set("pos", array_to_string(round_peg_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(square_peg_xml.find("./geom").get("size"))
        peg2_size = string_to_array(round_peg_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.nuts,
        )

    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            peg1_id = self.sim.model.body_name2id("peg1")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[peg1_id])

            name = "peg1_pos"
            observables[name] = Observable(
                name=name,
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )

        return observables


class Square_D2(Square_D1):
    """
    Even broader range for everything, and z-rotation randomization for peg.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

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

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0.25, 0.25),
                y=(-0.25, 0.25),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg=dict(
                x=(-0.25, 0.25),
                y=(-0.25, 0.25),
                z_rot=(0., np.pi / 2.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

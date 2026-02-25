# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script that offers an easy way to test random actions in a MimicGen environment.
Similar to the demo_random_action.py script from robosuite.
"""
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import numpy as np


def choose_mimicgen_environment():
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """

    # try to import robosuite task zoo to include those envs in the robosuite registry
    try:
        import robosuite_task_zoo
    except ImportError:
        print("Failed to import robosuite_task_zoo")
        pass

    # all base robosuite environments (and maybe robosuite task zoo)
    robosuite_envs = set(suite.ALL_ENVIRONMENTS)

    # all environments including mimicgen environments
    import mimicgen
    all_envs = set(suite.ALL_ENVIRONMENTS)

    # get only maze envs
    only_maze = sorted([env for env in all_envs if "Maze" in env or "Box" in env])
    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(only_maze):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(only_maze) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(only_maze) - 1)
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(only_maze[k]))

    print("Chosen environment: {}\n".format(only_maze[k]))
    # Return the chosen environment name
    return only_maze[k]


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment
    options["env_name"] =  choose_mimicgen_environment()

    # Choose robot
    options["robots"] = choose_robots(exclude_bimanual=True)

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="birdview",
        control_freq=20,
    )
    env.reset()

    # Get the camera ID for the camera you want to modify (e.g., "topview")
    camera_id = env.sim.model.camera_name2id("birdview")
    env.viewer.set_camera(camera_id=0)


    # Set the new camera position (x, y, z)
    env.sim.model.cam_pos[camera_id] = np.array([0, 0, 2])  # Modify these values to set the desired position
    env.sim.model.cam_quat[camera_id] = np.array([0.0, 0, 0, 1])  # Example quaternion
    #detector = Coffee_Detector(env)

    # Get action limits
    low, high = env.action_spec
    print("Action low limits:", low)
    print("Action high limits:", high)

    # do visualization
    prev_obs = env.reset()
    print("Available observation keys:", list(prev_obs.keys()))
    sum_displacement = 0
    for i in range(10000):
        action = np.zeros_like(low)
        curr_obs, reward, done, _ = env.step(action)
        # Use robot0_eef_pos instead of gripper1_pos
        displacement = np.linalg.norm(curr_obs['robot0_eef_pos'] - prev_obs['robot0_eef_pos'])
        sum_displacement += displacement
        print("Action:", action)
        print("Displacement:", displacement)
        prev_obs = curr_obs
        #detector.exclusively_occupying_gripper('coffee_pod')
        env.render()
    print("Total displacement:", sum_displacement)
    print("Average displacement per step:", sum_displacement / 10000)
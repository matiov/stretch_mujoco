robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}

depth_limits = {"d405": 1, "d435i": 10}


base_motion = {"timeout": 15, "default_x_vel": 0.3, "default_r_vel": 1.0}

# Default free-camera pose for the Mujoco UI viewer.
# Tune these values if you want a different startup viewpoint.
viewer_camera = {
    "lookat": [2.2, 0.6, -2.2],
    "distance": 12.3,
    "azimuth": 90.0,
    "elevation": -46.0,
}

# Print free-camera values while manually moving the Mujoco viewer camera.
viewer_camera_debug = {
    "enabled": False,
    # Minimum change needed before printing a new pose.
    "position_epsilon": 0.01,
    "angle_epsilon": 0.5,
}

# Optional viewer window placement for split-screen style layouts.
viewer_window = {
    "enabled": True,
    "layout": "left-half",  # supported: "left-half", "right-half"
    "max_attempts": 60,
}

REPLACEMENTS = {
    "obj_main": "water_bottle",
    "distr_counter_main": "coffee_cup",
    "coffee_machine_left_group_main": "coffee_machine",
    "object1": "blue_box",
    "object2": "red_cylinder",
}

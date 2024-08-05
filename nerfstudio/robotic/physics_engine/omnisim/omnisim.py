# load object and robotic trajectory from omnisim issac backend tensor


import numpy as np
import yaml
from pathlib import Path

from issac2sim import interpolate_trajectory_robotic_arm, interpolate_trajectory_robotic_arm_traj
import open3d as o3d
import torch
import trimesh

# the current grasp box case, I have a txt that save the robotic arm trajectory and the position of box

# and the indicator of the grasp start, success,fail, release four stages, reletive to the render stage information


def load_object_trajectory():
    """
    Load object trajectory from omnisim issac backend tensor
    """


    # bos pos and id from omnisim.py




    pass


def load_robotic_trajectory():
    """
    Load robotic trajectory from omnisim issac backend tensor
    """


    # robotic trajectory from omnisim.py
    pass

def parsing_grasp_stage(grasp_stage_file: Path):    
    """
    args:
        grasp_stage_file: Path to the grasp stage file
    
    returns:
        int: Status code (0 for success)
    
    This function parses the grasp stage file and sets the stage attributes
    
    """

    pass


def setup_params(meta_sim_path: Path):
    """
    Setup the parameters for simulation
    
    Args:
        meta_sim_path: Path to the meta simulation file
        config_params: Dictionary containing configuration parameters
    
    Returns:
        int: Status code (0 for success)
    """
    # Iterate over the items in config_params and set them as attributes
    config_params= yaml.load(meta_sim_path, Loader=yaml.FullLoader)
    for key, value in config_params.items():
        setattr(self, key, value)
    return 0
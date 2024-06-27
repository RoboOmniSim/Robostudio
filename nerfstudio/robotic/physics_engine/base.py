# define the category and physics engine relationship

import numpy as np
import torch
import yaml
import os

from dataclasses import dataclass, field

def load_config(config_path):

    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def semantic_category_engine_config(semantic_category,engine_id,config_file):
    """
    The engine backend are omnisim or python

    we choose the related engine based on the semantic category

    This config is aims to map the category and physics engine relationship
    The classic binding should be like:
    robotic arm: kinematic engine
    robotic gripper: gripper engine
    rigid object: newton euler engine
    soft object: FEM engine
    plastic object: MPM engine
    fluid object: SPH engine
    articulated object: articulated engine
        

    Args:
        semantic_category: the category of the object
        engine_id: the engine id
        config_file: the configuration file for the engine

    Returns:
        updated engine_id and config_file
        """

    return engine_id,config_file

def engine_config(raw_config,enable_backward=False):
    """
    The engine backend are omnisim or python


    This config is aims to map the category and physics engine relationship
    The classic binding should be like:
    robotic arm: kinematic engine
    robotic gripper: gripper engine
    rigid object: newton euler engine
    soft object: FEM engine
    plastic object: MPM engine
    fluid object: SPH engine
    articulated object: articulated engine
    

    Extra feature:
    energy conservation
    friction
    collision


    Args:
        raw_config: raw configuration file for omnisim
        enable_backward: enable backward for simulation based on monocular video


    Returns:

    """

    return 0



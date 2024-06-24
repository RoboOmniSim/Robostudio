# define the category and physics engine relationship

import numpy as np
import torch
import yaml
import os

from dataclasses import dataclass, field





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



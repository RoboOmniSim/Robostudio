



"""Config used for produce urdf file"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from nerfstudio.configs.base_config import InstantiateConfig, LoggingConfig, MachineConfig, ViewerConfig,PrintableConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.utils.rich_utils import CONSOLE
import numpy as np



@dataclass
class Urdfconfig(PrintableConfig):
    """Full config contents for running an experiment. Any experiment types (like training) will be
    subclassed from this, and must have their _target field defined accordingly."""

  
    center_vector: Optional[np.ndarray] = np.array([0,0,0])
    """center vector for deform ply"""

    scale_factor: Optional[np.ndarray] = np.array([0,0,0])
    """scale factor for deform ply"""

   
    add_gripper: Optional[bool] = False
    """whether to add gripper"""
    flip_x_coordinate: Optional[bool] = False
    """whether to flip x coordinate"""
    flip_y_coordinate: Optional[bool] = False
    """whether to flip y coordinate"""
    flip_z_coordinate: Optional[bool] = False
    """whether to flip z coordinate"""
    add_grasp_control: Optional[bool] = False
    """whether to add grasp control"""
    add_grasp_object: Optional[bool] = False
    """whether to add grasp object"""
   

    # urdf config

    gripper_model: Optional[str] = "default" # default or custom 
    """gripper model for urdf file"""

    arm_model: Optional[str] = "default" # default or custom
    """arm model for urdf file"""
    relationship_config_path: Optional[Path] = Path("config_info/relationship.yaml")
    """semantic simulation relationship config path for urdf file"""
    # Physics engine config
    novel_time: Optional[bool] = False
    """whether to add novel time interpolation"""
    link_edit_info: Optional[np.ndarray] = np.zeros(13)
    """link edit information for novel time interpolation"""

    engine_backend: Optional[str] = "python" # omni or python 
    """the backend of physics engine: omni is omnisim, python is gradsim based simulation"""

    assigned_ids: Optional[np.ndarray] = np.zeros(7)

    """the semantic id for group tracing and simulation"""

    semantic_category: Optional[np.ndarray] = np.zeros(7)
    """assign the semantic category for each group"""

    a : Optional[np.ndarray] = np.zeros(7)
    """assign the a value for each group"""

    alpha : Optional[np.ndarray] = np.zeros(7)
    """assign the alpha value for each group"""

    d : Optional[np.ndarray] = np.zeros(7)
    """assign the d value for each group"""

    joint_angles_degrees : Optional[np.ndarray] = np.zeros(7)
    """assign the joint angles for each group"""

    base_gt_scale : Optional[np.ndarray] = np.zeros(3)
    """assign the base ground truth scale in meters"""

    def setup_params(self, meta_sim_path: Path):
        """
        Setup the parameters for simulation
        
        Args:
            meta_sim_path: Path to the meta simulation file
            config_params: Dictionary containing configuration parameters
        
        Returns:
            int: Status code (0 for success)
        """
        # Iterate over the items in config_params and set them as attributes
        with open(meta_sim_path, 'r') as file:
            config_params = yaml.safe_load(file)
        for key, value in config_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid parameter")

        return 0
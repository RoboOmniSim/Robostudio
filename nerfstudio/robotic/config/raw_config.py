

"""Config used for running an experiment"""

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
class Roboticconfig(PrintableConfig):
    """Full config contents for running an experiment. Any experiment types (like training) will be
    subclassed from this, and must have their _target field defined accordingly."""

    output_dir: Path = Path("outputs")
    """relative or absolute output directory to save all checkpoints and logging"""
    experiment_type: Optional[str] = None
    """experiement name. Required to set in declarition"""
    project_name: Optional[str] = "robostudio_project"
    """Project name."""
    timestamp: str = "{timestamp}"
    """Experiment timestamp."""

    data: Optional[Path] = None
    """Alias for --pipeline.datamanager.data"""


    """command for export semantic ply"""
    expand_bbox: Optional[bool] = False
    """during export semantic ply, whether to expand bounding box to eliminate floater"""

    use_gripper: Optional[bool] = False
    """whether to use gripper during export semantic ply"""

    contain_object: Optional[bool] = False
    """whether to contain object during export semantic ply"""

    """ the command for export deform ply under certain dataset,"""

    center_vector: Optional[np.ndarray] = np.array([0,0,0])
    """center vector for deform ply"""

    scale_factor: Optional[np.ndarray] = np.array([0,0,0])
    """scale factor for deform ply"""

    simulation_timestamp: Optional[float] = 0
    """timestamp for object simulation"""

    add_simulation : Optional[bool] = False
    """whether to add objct simulation"""
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
    max_gripper_degree: Optional[float] = 0
    """max gripper degree"""

    """below is the render utils"""
    add_trajectory: Optional[bool] = False
    """whether to add trajectory"""
    render_camera_index: Optional[int] = 0
    """camera index for render"""

    start_time: Optional[float] = 0
    """start time for object simulation"""
    end_time: Optional[float] = 0
    """end time for object simulation"""

    end_time_collision: Optional[float] = 0
    """end time for object grasp simulation"""

    push_time_list_start:Optional[int] = 0
    """push time list for object simulation"""
    push_time_list_end:Optional[int] = 0
    """push time list for object simulation"""

    grasp_inter_time: Optional[float] = 0
    """grasp interval time for four different gripper simulation stage"""
    grasp_time_list_start: Optional[int] = 0
    """grasp time list for  gripper simulation stage 1, this is stage gripper close and move with object"""
    grasp_time_list_end: Optional[int] = 0
    """grasp time list for  gripper simulation stage 1, this is stage gripper close and move with object"""

    grasp_time_list_stage_2_start: Optional[int] = 0
    """grasp time list for  gripper simulation stage 2, this is stage gripper grasp success and move with object"""
    grasp_time_list_stage_2_end: Optional[int] = 0
    """grasp time list for  gripper simulation stage 2, this is stage gripper grasp success and move with object"""

    grasp_time_list_stage_3_start: Optional[int] = 0
    """grasp time list for  gripper simulation stage 3, this is stage gripper release object"""
    grasp_time_list_stage_3_end: Optional[int] = 0
    """grasp time list for  gripper simulation stage 3, this is stage gripper release object"""

    time_list_start: Optional[int] =0
    """time list for rendering"""
    time_list_end: Optional[int] = 0
    """time list for rendering"""
    novel_fps_rate: Optional[float] = 0
    """novel fps rate for rendering"""

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

    semantic_category: Optional[list] = ["kinematic","gripper","newton_euler","FEM","MPM","SPH","articulated"]
    """assign the semantic category for each group"""


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

# helper fucntion, load omnisim metaconfig and transfer it to nerfstudio config


def omni2gs_config(omni_config_path: Path, gs_config_path: Path):
    """
    Transform the omnisim metaconfig to nerfstudio config
    
    Args:
        omni_config_path: Path to the omnisim metaconfig
        gs_config_path: Path to the nerfstudio config
    
    return:

    """
    
    # 


    return 0



# helper function, export urdf file to omnisim conifg

def export_urdf_to_omnisim_config(omni_urdf_path: Path, twodgs_urdf_config_path: Path):
    """
    export scale, center vector, part mesh, and llm infered parameter to omnisim config
    
    Args:
        omni_urdf_path: Path to the omnisim urdf file
        twodgs_urdf_config_path: Path to the 2dgs config
    
    """

    ## load urdf file from pre-build urdf file or from llm generated urdf file


    ## rebuild mesh by part



    ## export all information to omnisim config



    # object level urdf : mass, friction, edit urdf's dae and obj from y-up to z-up, edit parameter in dae, edit rpy to pi/2,0,0 in obj urdf


    # setup target position of gripper center



    # table scale


    # real world position
    #franka_pose.p = gymapi.Vec3(0, 0, 0.28)

    #table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

    return 0




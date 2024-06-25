# load object and robotic trajectory from omnisim issac backend tensor


import numpy as np
import yaml
from pathlib import Path



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
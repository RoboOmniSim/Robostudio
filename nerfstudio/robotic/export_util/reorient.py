import torch
import pypose as pp
import numpy as np





def recenter(pose, global_translation):

    """
    recenter camera and scenes
    
    """

    pose[:3] += global_translation



    return pose
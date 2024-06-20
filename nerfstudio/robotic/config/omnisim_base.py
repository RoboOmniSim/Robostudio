import numpy as np

import yaml
import math
import sys
import os
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))




class MetaConfig :

    def __init__(self) :

        pass

    def update(self, another) :

        for key in another.keys() :
            setattr(self, key, another[key])



class MetaGS(MetaConfig):

    # this is for pass parameter from gs to omnisim

    def __init__(self, config : dict):

        # 

        self.width = config["width"]
        self.height = config["height"]
        self.horizontal_fov = config["horizontal_fov"]
        self.enable_tensors = config["enable_tensors"]
        self.camera_locate_pos = config["camera_locate_pos"]
        self.camera_target_pos = config.get("camera_target_pos", None)
        self.camera_rot = config.get("camera_rot", None)

        self.check()
    
    def check(self) :

        if self.camera_target_pos != None and self.camera_rot != None:

            print("Warning: camera target and rot are both set, target will be ignored")
        
        if self.camera_target_pos == None and self.camera_rot == None:

            print("Warning: camera target and rot are both not set, default target will be used")
            self.camera_target_pos = [0, 0, 0]

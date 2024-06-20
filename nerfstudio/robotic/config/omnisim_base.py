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

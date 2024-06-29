# define the category and physics engine relationship

import numpy as np
import torch
import yaml
import os

from dataclasses import dataclass, field

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['categories']

# Function to get the engine based on category
def get_engine_by_category(categories, category):
    return categories.getattr(category)



def assign_id_based_on_engine_name(engine_name):
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
        

    Args:
        engine_name: the name of the engine

    Returns:
        engine_id: the id of the engine
    """

    engine_inference=np.zeros(len(engine_name))
    
    for i in range(len(engine_name)):
        if engine_name[i]=="kinematic":
            engine_inference[i]=1
        elif engine_name[i]=="gripper":
            engine_inference[i]=2
        elif engine_name[i]=="newton_euler":
            engine_inference[i]=3
        elif engine_name[i]=="FEM":
            engine_inference[i]=4
        elif engine_name[i]=="MPM":
            engine_inference[i]=5
        elif engine_name[i]=="SPH":
            engine_inference[i]=6
        elif engine_name[i]=="articulated":
            engine_inference[i]=7
        else:
            engine_inference[i]=0

    return engine_inference


def semantic_category_engine_config(semantic_category,config_file):
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
        config_file: the configuration file for the engine

    Returns:
        updated engine_id and config_file # int that shares the same length as the semantic category
        """
    engine=get_engine_by_category(config_file,semantic_category)

    # aasign the engine id based on the engine

    engine_inference=assign_id_based_on_engine_name(engine)


    return engine_inference




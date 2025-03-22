
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import trimesh



from nerfstudio.robotic.physics_engine.python.collision_detection import collision_detection
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.kinematic.control_helper import *

def load_gripper_control(output_file,experiment_type,start_time=100,end_time=200):
    """
    Load the gripper control data from the given path.
    """
    if experiment_type=='push_bag':
        # Load the gripper control data for the push box experiment
        movement_angle_state = read_txt_file(output_file) # radians
    elif experiment_type=='novelpose':
        # Load the gripper control data for the pick and place experiment
        movement_angle_state = read_txt_file(output_file) # radians
    elif experiment_type=='grasp':
        # Load the gripper control data for the open and close experiment
        movement_angle_state = read_txt_file(output_file) # radians
    elif experiment_type=='grasp_object':
        # Load the gripper control data for the push box experiment
        movement_angle_state = read_txt_file(output_file) # radians
    else:
        # Load the gripper control data for the default experiment
        movement_angle_state = read_txt_file(output_file) # radians

    joint_angles_degrees_gripper = np.zeros(5)
    a_gripper = np.zeros(5)
    alpha_gripper = np.zeros(5)
    d_gripper = np.zeros(5)


    joint_angles_degrees_gripper[1]=1.83759# for left down
    joint_angles_degrees_gripper[2]=2.8658# for left up
    joint_angles_degrees_gripper[3]=1.83759# for right down
    joint_angles_degrees_gripper[4]=2.8658# for right up

    a_gripper[0]=0
    a_gripper[1]=-0.03853 
    a_gripper[2]=0.041998
    a_gripper[3]=-0.03853
    a_gripper[4]=0.041998

    alpha_gripper[1]=np.pi/2
    alpha_gripper[2]=0
    alpha_gripper[3]=np.pi/2
    alpha_gripper[4]=0

    d_gripper[0]=0.11
    d_gripper[1]=0
    d_gripper[2]=0
    d_gripper[3]=0
    d_gripper[4]=0

    gripper_control_mode = np.zeros(len(movement_angle_state))


    dof_limit=20
    degrees = np.linspace(0, dof_limit, num=int((end_time-start_time)*10))  # 10 fps

    # Convert degrees to radians
    radians = np.deg2rad(degrees)
    # write a linear interpolation from start angle to end angle
    for i in range(start_time,end_time):
        gripper_control_mode[i]=radians[i-start_time]
    # gripper_control_mode[100]=1 # the start time of gripper



    return gripper_control_mode,joint_angles_degrees_gripper, a_gripper, alpha_gripper, d_gripper









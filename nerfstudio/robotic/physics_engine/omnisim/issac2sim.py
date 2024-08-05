import os
import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
from nerfstudio.robotic.kinematic.uniform_kinematic import *
import torch
import argparse

from nerfstudio.robotic.physics_engine.python.novel_time import *

def extract_values(string_list):
    values = []
    for s in string_list:
        # Remove unwanted characters
        clean_s = s.replace('[', '').replace(']', '').strip()
        # Convert to float and add to the values list
        values.append(float(clean_s))
    return values

def load_trajectory(path):
    with open(path, 'r') as file:
        data = file.read()
    
    # Split the data by 'Dof Positions:'
    blocks = data.split('Dof Positions:')[1:]
    
    # Parse each block of data
    dof_positions = []
    for block in blocks:
        # Extract the values from the block
        values = block.strip().strip('[]').split('], [')
        positions = [list(extract_values(value.split(','))) for value in values]
        dof_positions.append(positions)
    
    return dof_positions

def linear_interpolation(start, end, steps):
    return np.array([start + (end - start) * i / steps for i in range(steps)])
    

def interpolate_trajectory_robotic_arm(trajectory_file, steps, joint_num=6, default_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6','gripper_main','gripper_left_down', 'gripper_left_up', 'gripper_right_down', 'gripper_right_up'],link_edit=np.zeros(13)):
    """
    Args:
    trajectory_file (str): Path to the trajectory file.
    steps (int): Number of steps to interpolate between each pair of joint angles.(for example 6 from 10 fps to 60 fps)
    joint_num (int): Number of joints in the robotic arm.
    default_joint_names (list): List of joint names.
    
    
    """
    
    # resize the movement_angle_state to the same length as the timestamp of trajectory 
    traj = np.array(load_trajectory(trajectory_file)).flatten().reshape(-1, joint_num) # for six dof path
    adaptive_length = len(traj)
    traj_mode_zeros  = [
                    {
                        "Time": np.zeros(1),
                        "Joint Names": default_joint_names,
                        "Joint Positions":  np.zeros(11) # for 6dof plus 5 gripper joints
                    }
                    for _ in range(adaptive_length)
                    ]
    
    positions=traj["Joint Positions"]
    expand_length=adaptive_length*steps
    interpolated_angles_list=[]
    time_list=[]

    for i in range(len(traj)):
        

        interpolated_angles=linear_interpolation_angle(positions[i], positions[i+1], steps)
        interpolated_angles_list.append(interpolated_angles)
        time_list.append(np.linspace(i, i+1, steps))
    traj_mode_zeros["Time"]=time_list
    traj_mode_zeros["Joint Positions"]=interpolated_angles_list

    return traj_mode_zeros



def interpolate_trajectory_robotic_arm_traj(trajectory_file, steps, traj_mode,joint_num=6, default_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6','gripper_main','gripper_left_down', 'gripper_left_up', 'gripper_right_down', 'gripper_right_up'],link_edit=np.zeros(13)):
    """
    Args:
    trajectory_file (str): Path to the trajectory file.
    steps (int): Number of steps to interpolate between each pair of joint angles.(for example 6 from 10 fps to 60 fps)
    joint_num (int): Number of joints in the robotic arm.
    default_joint_names (list): List of joint names.
    
    
    """
    
    # resize the movement_angle_state to the same length as the timestamp of trajectory 
    # traj = np.array(load_trajectory(trajectory_file)).flatten().reshape(-1, joint_num) # for six dof path
    traj=traj_mode
    adaptive_length = len(traj)
    traj_mode_zeros  = [
                    {
                        "Time": np.zeros(1),
                        "Joint Names": default_joint_names,
                        "Joint Positions":  np.zeros(11) # for 6dof plus 5 gripper joints
                    }
                    for _ in range(adaptive_length)
                    ]
    
    positions=traj_mode["Joint Positions"]
    expand_length=adaptive_length*steps
    interpolated_angles_list=[]
    time_list=[]

    for i in range(len(traj)):
        

        interpolated_angles=linear_interpolation_angle(positions[i], positions[i+1], steps)
        interpolated_angles_list.append(interpolated_angles)
        time_list.append(np.linspace(i, i+1, steps))
    traj_mode_zeros["Time"]=time_list
    traj_mode_zeros["Joint Positions"]=interpolated_angles_list

    return traj_mode

def edit_trajectory(trajectory_file, steps,joint_num=6,default_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6','gripper_main','gripper_left_down', 'gripper_left_up', 'gripper_right_down', 'gripper_right_up'],link_edit=np.zeros(13)):
    """
    enables trajectory editing like moveit
    
    
    """
       # resize the movement_angle_state to the same length as the timestamp of trajectory 
    traj = np.array(load_trajectory(trajectory_file)).flatten().reshape(-1, joint_num) # for six dof path
    adaptive_length = len(traj)
    traj_mode  = [
                    {
                        "Time": np.zeros(1),
                        "Joint Names": default_joint_names,
                        "Joint Positions":  np.zeros(11) # for 6dof plus 5 gripper joints
                    }
                    for _ in range(adaptive_length)
                    ]
    little_t=np.linspace(0, adaptive_length, adaptive_length)

    max_link_1=link_edit[1]
    max_link_2=link_edit[2]
    max_link_3=link_edit[3]
    max_link_4=link_edit[4]

    extra_link_1=linear_interpolation(0, max_link_1, 10)
    extra_link_2=linear_interpolation(0, max_link_2, 10)
    extra_link_3=linear_interpolation(0, max_link_3, 10)
    extra_link_4=linear_interpolation(0, max_link_4, 10)

    extra_link_1_reverse=-extra_link_1
    extra_link_2_reverse=-extra_link_2
    extra_link_3_reverse=-extra_link_3
    extra_link_4_reverse=-extra_link_4

    gripper_link_1=0.7

    stage_1=0
    stage_2=0
    stage_3=0
    stage_4=0
    for i in range(adaptive_length):
        traj_mode[i]["Time"] = little_t
        traj_mode[i]["Joint Positions"][:6] = traj[i]
        traj_mode[i]["Joint Positions"][6:] = [0,0,0,0,0] # for the gripper joints
                        
        if 230< i <= 240:

            # get close to object
            traj_mode[i]["Joint Positions"][1] = traj_mode[i]["Joint Positions"][1]+extra_link_1[stage_1]
            traj_mode[i]["Joint Positions"][2] = traj_mode[i]["Joint Positions"][2]+extra_link_2[stage_1]
            traj_mode[i]["Joint Positions"][3] = traj_mode[i]["Joint Positions"][3]+extra_link_3[stage_1] 
            traj_mode[i]["Joint Positions"][4] = traj_mode[i]["Joint Positions"][4]+extra_link_4[stage_1]
            stage_1+=1
        if 240< i <= 250:
            # gripper close and interact with object
            traj_mode[i]["Joint Positions"][1] = traj_mode[i]["Joint Positions"][1]+extra_link_1[stage_1-1]
            traj_mode[i]["Joint Positions"][2] = traj_mode[i]["Joint Positions"][2]+extra_link_2[stage_1-1]
            traj_mode[i]["Joint Positions"][3] = traj_mode[i]["Joint Positions"][3]+extra_link_3[stage_1-1] 
            traj_mode[i]["Joint Positions"][4] = traj_mode[i]["Joint Positions"][4]+extra_link_4[stage_1-1]
            stage_2+=1
        if 250< i <= 260:
            # grasp success object move with gripper
            traj_mode[i]["Joint Positions"][1] = traj_mode[i]["Joint Positions"][1]+extra_link_1[stage_1-1]+extra_link_1_reverse[stage_3]
            traj_mode[i]["Joint Positions"][2] = traj_mode[i]["Joint Positions"][2]+extra_link_2[stage_1-1]+extra_link_2_reverse[stage_3]
            traj_mode[i]["Joint Positions"][3] = traj_mode[i]["Joint Positions"][3]+extra_link_3[stage_1-1]+extra_link_3_reverse[stage_3]
            traj_mode[i]["Joint Positions"][4] = traj_mode[i]["Joint Positions"][4]+extra_link_4[stage_1-1]+extra_link_4_reverse[stage_3]
            stage_3+=1

    return traj_mode

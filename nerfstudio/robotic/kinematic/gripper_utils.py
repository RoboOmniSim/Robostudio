import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import trimesh



from nerfstudio.robotic.physics_engine.collision_detection import collision_detection
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.kinematic.control_helper import *






# Function to calculate all transformation matrices and the final matrix
def gripper_mdh(movement_angle_state_link,joint_angles_degrees_link, a_link, alpha_link, d_link,
                gripper_control,joint_angles_degrees_gripper, a_gripper, alpha_gripper, d_gripper,i=0,flip_x_coordinate=False,add_grasp_control=0):
    """
    calulate transformation matrices based dh parameters

    Args:
    i: timestamp
    joint_angles_degrees: list of joint angles in degrees
    a: list of a parameters
    alpha: list of alpha parameters
    d: list of d parameters

    # the dimension of control with gripper means 6+5


    # the 7 shares the same information with 6


    # 8 is the left_down
    # 9 is the left_up
    # 10 is the right_down
    # 11 is the right_up
    
    
    """
    state_i = movement_angle_state_link[i]['Time']
    state_name=movement_angle_state_link[i]['Joint Names']
    state_position=movement_angle_state_link[i]['Joint Positions']
    joint_angle_deform_link= np.array(state_position)
    if a_gripper == None:
        joint_angles_degrees=joint_angles_degrees_link
        a=a_link
        alpha=alpha_link
        d=d_link
        joint_angle_deform=joint_angle_deform_link

    else:
        joint_angles_degrees=np.concatenate((joint_angles_degrees_link,joint_angles_degrees_gripper),axis=0)
        a=np.concatenate((a_link,a_gripper),axis=0)
        alpha=np.concatenate((alpha_link,alpha_gripper),axis=0)
        d=np.concatenate((d_link,d_gripper),axis=0)
        joint_angle_deform_gripper=np.zeros((5)) # 5 is the number of gripper joint
                # add the value here 
        joint_angle_deform_gripper=np.zeros((5)) # 5 is the number of gripper joint
        joint_angle_deform_gripper[1]= np.array(gripper_control[i]) # just for test 
        joint_angle_deform_gripper[3]= np.array(gripper_control[i])
        joint_angle_deform=np.concatenate((np.round(joint_angle_deform_link, 3),np.round(joint_angle_deform_gripper, 3)),axis=0)

    joint_angles_radians_raw = joint_angles_degrees 
    


    # print('joint_angle_deform' ,joint_angle_deform)

    gripper_deform=add_grasp_control  # 0 to -0.8525
    if i !=0:
        joint_angle_deform[0]=joint_angle_deform[0]
        joint_angle_deform[1]=joint_angle_deform[1]
        joint_angle_deform[2]=joint_angle_deform[2]
        joint_angle_deform[3]=joint_angle_deform[3]
        joint_angle_deform[4]=joint_angle_deform[4]
        joint_angle_deform[5]=joint_angle_deform[5]
        joint_angle_deform[6]=0
        joint_angle_deform[7]=gripper_deform
        joint_angle_deform[8]=gripper_deform*-1 # invert with 7
        joint_angle_deform[9]=gripper_deform*-1# invert with 7
        joint_angle_deform[10]=gripper_deform# invert with 9
        
        
    # print('reshape joint_angle_deform' ,joint_angle_deform)
    joint_angles_radians=joint_angles_radians_raw+joint_angle_deform
    # print('joint_angles_radians',joint_angles_radians) 

    transformations = []
    j=0

    gripper_index_list=[7,9]
    for theta, a_i, alpha_i, d_i in zip(joint_angles_radians, a, alpha, d): # 11 now, adapt the linkage for different connenciton
        
        # apply edit gripper mdh for the control left down gripper and right down gripper
        if j==9:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        
        elif j==7:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        else:
            T_temp=create_transformation_matrix_mdh(theta, a_i, alpha_i, d_i)

        if flip_x_coordinate:
            if j not in gripper_index_list:
                T_temp=reflect_x_axis(T_temp) # for different base coordinate, the default base coordinate is - + - 
            else:
                # T_temp=create_transformation_matrix_mdh_gripper_reflect_x_coordinate(theta, a_i, alpha_i, d_i) # for different base coordinate, the default base coordinate is - + - since mdh gripper is different from mdh
                T_temp=reflect_x_axis_only(T_temp)
        # T_temp=create_transformation_matrix_mdh(theta, a_i, alpha_i, d_i)
        # if flip_x_coordinate:
        #     T_temp=reflect_x_axis(T_temp) # for different base coordinate, the default base coordinate is - + - 
        j+=1
        # for gripper right 10,11, it is connect to 7 
        transformations.append(T_temp)
    
    # Calculate the final transformation from the base to the end-effector
    final_transformation = np.eye(4)
    final_transformations_list=[[]]*len(transformations)

    p=0
    for transformation in transformations:
            if p==9:
                gripper_move= final_transformations_list[6]
                final_transformation = np.dot(gripper_move, transformation)
                final_transformations_list[p]=final_transformation
                p+=1
            elif p==10:
                gripper_right_move= final_transformations_list[9]
                final_transformation = np.dot(gripper_right_move, transformation)
                final_transformations_list[p]=final_transformation
                p+=1
            else:
                final_transformation = np.dot(final_transformation, transformation)
                final_transformations_list[p]=final_transformation
                p+=1
    return transformations, final_transformations_list




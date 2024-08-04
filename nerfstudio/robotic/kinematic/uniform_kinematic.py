import numpy as np
import torch
import open3d as o3d
import trimesh

import os

from nerfstudio.robotic.kinematic.gripper_utils import reflect_x_axis,reflect_y_axis,reflect_z_axis

from nerfstudio.robotic.kinematic.control_helper import *



# part 1
def load_uniform_kinematic(output_file_path,experiment_type,scale_factor_pass=np.zeros((3,1)),center_vector_pass=np.zeros((3,1)),add_gripper=False,flip_x_coordinate=False):
        movement_angle_state = read_txt_file(output_file_path) # radians

        name=experiment_type
        scale_factor=np.zeros((3))
        scale_factor_pass=np.array(scale_factor_pass)
        center_vector_pass=np.array(center_vector_pass)
        center_vector_gt=center_vector_pass
        scale_factor=scale_factor_pass

        if name=="issac2sim":
            a = [0, 0, -0.427, -0.357, 0, -0.015] # pass the gripper a value to the uniform kinematic
        elif name=="push_box":
            a = [0, 0, -0.427, -0.357, 0, -0.015]
        elif name=="novel_pose":
            a = [0, 0, -0.427, -0.357, 0, 0] # gripper close and other case
        elif name=="grasp":
            a = [0, 0, -0.427, -0.357, 0, 0] # gripper close and other case
        elif name=="grasp_object":
            a = [0, 0, -0.427, -0.357, 0, 0] # gripper close and other case
        #alpha should be default from urdf
        else:
            a = [0, 0, -0.427, -0.357, 0, 0]

        alpha = [0, np.pi/2, 0.07, 0, np.pi/2, -np.pi/2] # this 0.07 only for manual adjust for the novel pose because of the numerical error in radians degree and sin computation
        # alpha = [0, np.pi/2, 0, 0, np.pi/2, -np.pi/2] # original


        d = [0.147, 0, 0.025, 0.116, 0.116, 0.105] # this 0.025 only for manual adjust for the novel pose because of the numerical error in radians degree and sin computation
        # d = [0.147, 0, 0, 0.116, 0.116, 0.105] # original
        joint_angles_degrees = [0, -np.pi/2, 0, -np.pi/2, 0, 0]  # Update with actual angles from the example

        # use original urdf to compute a,d,scale 

        scale_a=np.array([1,1,1,1,1,1])/scale_factor[2]  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
        scale_d=np.array([1,1,1,1,1,1])/scale_factor[2] #  s1 is x axis for d1 ,s4 and s6  is y axis 



        scale_d[3]=1/scale_factor[1]
        scale_d[5]=1/scale_factor[1]

        a=a*scale_a
        d=d*scale_d
        
        if add_gripper:
            joint_angles_degrees_gripper = np.zeros(5)
            a_gripper = np.zeros(5)
            alpha_gripper = np.zeros(5)
            d_gripper = np.zeros(5)



            # MODIFIED  
            joint_angles_degrees_gripper[1]=1.83759# for left down
            joint_angles_degrees_gripper[2]=2.8658# for left up
            joint_angles_degrees_gripper[3]=1.30405# for right down
            joint_angles_degrees_gripper[4]=-2.8658# for right up

            if name=="issac2sim":
                a_gripper[0]=-0.0 # pass the gripper a value to the uniform kinematic
            elif name=="push_box":
                a_gripper[0]=-0.0
            elif name=="novel_pose":
                a_gripper[0]=-0.025 # gripper close case
            elif name=="grasp":
                a_gripper[0]=-0.025 # gripper close case
            elif name=="grasp_object":
                a_gripper[0]=-0.025 # gripper close case
            else:
                a_gripper[0]=-0.025 # gripper close case
            
            a_gripper[1]=-0.03853/scale_factor[2]# gt from phyiscs parameter
            a_gripper[2]=0.041998/scale_factor[2]
            # we apologize for this due to the numerical error, 
            #the scale is really a huge gap from meters to mm so minor error may raise, we will fixe it in future
            a_gripper[3]=0.03853*0.6
            a_gripper[4]=0.041998/scale_factor[2]

            
            alpha_gripper[1]=np.pi/2
            alpha_gripper[2]=0
            alpha_gripper[3]=np.pi/2
            alpha_gripper[4]=0

            d_gripper[0]=0.11035/scale_factor[1] # gt from phyiscs parameter
            # d_gripper[0]=0
            d_gripper[1]=0
            d_gripper[2]=0
            d_gripper[3]=0
            d_gripper[4]=0

            gripper_control_mode  = [
                {
                    "Time": np.zeros(1),
                    "Joint Names": ['gripper_main','gripper_left_down', 'gripper_left_up', 'gripper_right_down', 'gripper_right_up'],
                    "Joint Positions": 0
                }
                for _ in range(len(movement_angle_state))
            ]

            # dof_limit=20
            degrees = np.zeros_like(d_gripper)  # 10 fps

            # Convert degrees to radians
            radians = np.deg2rad(degrees)
            # write a linear interpolation from start angle to end angle
            for i in range(len(movement_angle_state)):
                gripper_control_mode[i]['Joint Positions']=radians
            joint_angles_degrees=np.concatenate((joint_angles_degrees,joint_angles_degrees_gripper),axis=0)
            a=np.concatenate((a,a_gripper),axis=0)
            alpha=np.concatenate((alpha,alpha_gripper),axis=0)
            d=np.concatenate((d,d_gripper),axis=0)
            if len(gripper_control_mode) != len(movement_angle_state):
                raise ValueError("Lists must have the same length")

            # Concatenate the lists
            combined_list = []
            for i in range(len(gripper_control_mode)):
                combined_entry = {
                    "Time":np.concatenate((np.array(movement_angle_state[i]["Time"]).reshape(1), gripper_control_mode[i]["Time"]),axis=0) ,
                    "Joint Names": movement_angle_state[i]["Joint Names"] + gripper_control_mode[i]["Joint Names"],
                    "Joint Positions": np.concatenate((movement_angle_state[i]["Joint Positions"],gripper_control_mode[i]["Joint Positions"]),axis=0)
                }
                combined_list.append(combined_entry)
            movement_angle_state=combined_list

            individual_transformations_0, final_transformations_list_0 = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0,add_gripper=add_gripper,flip_x_coordinate=flip_x_coordinate)
        else:    
            individual_transformations_0, final_transformations_list_0 = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0)

        return movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt



# version 2
def load_uniform_kinematic_v2(output_file_path,experiment_type,scale_factor_pass=np.zeros((3,1)),center_vector_pass=np.zeros((3,1)),add_gripper=False,flip_x_coordinate=False,flip_y_coordinate=False,flip_z_coordinate=False,gripper_model="default",arm_model="default"):
        movement_angle_state = read_txt_file(output_file_path) # radians

        name=experiment_type
        scale_factor=np.zeros((3))
        scale_factor_pass=np.array(scale_factor_pass)
        center_vector_pass=np.array(center_vector_pass)
        center_vector_gt=center_vector_pass
        scale_factor=scale_factor_pass

        if name=="issac2sim":
            a = [0, 0, -0.427, -0.357, 0, -0.015] # pass the gripper a value to the uniform kinematic
        elif name=="push_box":
            a = [0, 0, -0.427, -0.357, 0, -0.015]
        elif name=="novel_pose":
            a = [0, 0, -0.427, -0.357, 0, 0] # gripper close and other case
        elif name=="grasp":
            a = [0, 0, -0.427, -0.357, 0, 0] # gripper close and other case
        elif name=="grasp_object":
            a = [0, 0, -0.427, -0.357, 0, 0] # gripper close and other case
        #alpha should be default from urdf
        else:
            a = [0, 0, -0.427, -0.357, 0, 0]

        alpha = [0, np.pi/2, 0.07, 0, np.pi/2, -np.pi/2] # this 0.07 only for manual adjust for the novel pose because of the numerical error in radians degree and sin computation
        # alpha = [0, np.pi/2, 0, 0, np.pi/2, -np.pi/2] # original


        d = [0.147, 0, 0.025, 0.116, 0.116, 0.105] # this 0.025 only for manual adjust for the novel pose because of the numerical error in radians degree and sin computation
        # d = [0.147, 0, 0, 0.116, 0.116, 0.105] # original
        joint_angles_degrees = [0, -np.pi/2, 0, -np.pi/2, 0, 0]  # Update with actual angles from the example

        # use original urdf to compute a,d,scale 

        scale_a=np.array([1,1,1,1,1,1])/scale_factor[2]  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
        scale_d=np.array([1,1,1,1,1,1])/scale_factor[2] #  s1 is x axis for d1 ,s4 and s6  is y axis 



        scale_d[3]=1/scale_factor[1]
        scale_d[5]=1/scale_factor[1]

        a=a*scale_a
        d=d*scale_d
        
        if add_gripper:
            joint_angles_degrees_gripper = np.zeros(5)
            a_gripper = np.zeros(5)
            alpha_gripper = np.zeros(5)
            d_gripper = np.zeros(5)



            # MODIFIED  
            joint_angles_degrees_gripper[1]=1.83759# for left down
            joint_angles_degrees_gripper[2]=2.8658# for left up
            joint_angles_degrees_gripper[3]=1.30405# for right down
            joint_angles_degrees_gripper[4]=-2.8658# for right up

            if name=="issac2sim":
                a_gripper[0]=-0.0 # pass the gripper a value to the uniform kinematic
            elif name=="push_box":
                a_gripper[0]=-0.0
            elif name=="novel_pose":
                a_gripper[0]=-0.025 # gripper close case
            elif name=="grasp":
                a_gripper[0]=-0.025 # gripper close case
            elif name=="grasp_object":
                a_gripper[0]=-0.025 # gripper close case
            else:
                a_gripper[0]=-0.025 # gripper close case
            
            a_gripper[1]=-0.03853/scale_factor[2]# gt from phyiscs parameter
            a_gripper[2]=0.041998/scale_factor[2]
            # we apologize for this due to the numerical error, 
            #the scale is really a huge gap from meters to mm so minor error may raise, we will fixe it in future
            a_gripper[3]=0.03853*0.6
            a_gripper[4]=0.041998/scale_factor[2]

            
            alpha_gripper[1]=np.pi/2
            alpha_gripper[2]=0
            alpha_gripper[3]=np.pi/2
            alpha_gripper[4]=0

            d_gripper[0]=0.11035/scale_factor[1] # gt from phyiscs parameter
            # d_gripper[0]=0
            d_gripper[1]=0
            d_gripper[2]=0
            d_gripper[3]=0
            d_gripper[4]=0

            gripper_control_mode  = [
                {
                    "Time": np.zeros(1),
                    "Joint Names": ['gripper_main','gripper_left_down', 'gripper_left_up', 'gripper_right_down', 'gripper_right_up'],
                    "Joint Positions": 0
                }
                for _ in range(len(movement_angle_state))
            ]

            # dof_limit=20
            degrees = np.zeros_like(d_gripper)  # 10 fps

            # Convert degrees to radians
            radians = np.deg2rad(degrees)
            # write a linear interpolation from start angle to end angle
            for i in range(len(movement_angle_state)):
                gripper_control_mode[i]['Joint Positions']=radians
            joint_angles_degrees=np.concatenate((joint_angles_degrees,joint_angles_degrees_gripper),axis=0)
            a=np.concatenate((a,a_gripper),axis=0)
            alpha=np.concatenate((alpha,alpha_gripper),axis=0)
            d=np.concatenate((d,d_gripper),axis=0)
            if len(gripper_control_mode) != len(movement_angle_state):
                raise ValueError("Lists must have the same length")

            # Concatenate the lists
            combined_list = []
            for i in range(len(gripper_control_mode)):
                combined_entry = {
                    "Time":np.concatenate((np.array(movement_angle_state[i]["Time"]).reshape(1), gripper_control_mode[i]["Time"]),axis=0) ,
                    "Joint Names": movement_angle_state[i]["Joint Names"] + gripper_control_mode[i]["Joint Names"],
                    "Joint Positions": np.concatenate((movement_angle_state[i]["Joint Positions"],gripper_control_mode[i]["Joint Positions"]),axis=0)
                }
                combined_list.append(combined_entry)
            movement_angle_state=combined_list

            individual_transformations_0, final_transformations_list_0 = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0,add_gripper=add_gripper,flip_x_coordinate=flip_x_coordinate)
        else:    
            individual_transformations_0, final_transformations_list_0 = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0)

        return movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt





# Function to calculate all transformation matrices and the final matrix
def calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0,add_gripper=False,flip_x_coordinate=False):
    """
    calulate transformation matrices based dh parameters

    Args:
    i: timestamp
    joint_angles_degrees: list of joint angles in degrees
    a: list of a parameters
    alpha: list of alpha parameters
    d: list of d parameters

    
    
    
    """
    state_i = movement_angle_state[i]['Time']
    state_name=movement_angle_state[i]['Joint Names']
    state_position=movement_angle_state[i]['Joint Positions']


    joint_angles_radians_raw = joint_angles_degrees
    joint_angle_deform= np.array(state_position)
    joint_angle_deform=np.round(joint_angle_deform, 3)



    joint_angles_radians=joint_angles_radians_raw+joint_angle_deform


    transformations = []

    j=0
    gripper_index_list=[7,9]
    
    for theta, a_i, alpha_i, d_i in zip(joint_angles_radians, a, alpha, d):
        

        # apply edit gripper mdh for the control left down gripper and right down gripper
        if j==9:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        
        elif j==7:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        else:
            T_temp=create_transformation_matrix_mdh(theta, a_i, alpha_i, d_i)

        # gripper has different base coordinate so need new flip coordinate
        if flip_x_coordinate:
            if j not in gripper_index_list:
                T_temp=reflect_x_axis(T_temp) # for different base coordinate, the default base coordinate is - + - 
            else:
                # T_temp=create_transformation_matrix_mdh_gripper_reflect_x_coordinate(theta, a_i, alpha_i, d_i) # for different base coordinate, the default base coordinate is - + - since mdh gripper is different from mdh
                T_temp=reflect_x_axis_only(T_temp)
        # for gripper right 10,11, it is connect to 7 
        transformations.append(T_temp)

        j+=1
    # Calculate the final transformation from the base to the end-effector
    final_transformation = np.eye(4)
    final_transformations_list=[[]]*len(joint_angles_radians)

    p=0
    if add_gripper:

        
        for transformation in transformations:
            if p==9:
                gripper_move= final_transformations_list[6]
                final_transformation = np.dot(gripper_move, transformation)
                final_transformations_list[p]=final_transformation
            elif p==10:
                gripper_right_move= final_transformations_list[9]
                final_transformation = np.dot(gripper_right_move, transformation)
                final_transformations_list[p]=final_transformation
            else:
                final_transformation = np.dot(final_transformation, transformation)
                final_transformations_list[p]=final_transformation
            p+=1
    else:
        for transformation in transformations:
            final_transformation = np.dot(final_transformation, transformation)
            final_transformations_list[p]=final_transformation
            p+=1
    return transformations, final_transformations_list

def calculate_transformations_mdh_v2(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0,add_gripper=False,flip_x_coordinate=False,flip_y_coordinate=False,flip_z_coordinate=False):
    """
    calulate transformation matrices based dh parameters

    Args:
    i: timestamp
    joint_angles_degrees: list of joint angles in degrees
    a: list of a parameters
    alpha: list of alpha parameters
    d: list of d parameters

    
    
    
    """
    state_i = movement_angle_state[i]['Time']
    state_name=movement_angle_state[i]['Joint Names']
    state_position=movement_angle_state[i]['Joint Positions']


    joint_angles_radians_raw = joint_angles_degrees
    joint_angle_deform= np.array(state_position)
    joint_angle_deform=np.round(joint_angle_deform, 3)



    joint_angles_radians=joint_angles_radians_raw+joint_angle_deform


    transformations = []

    j=0
    gripper_index_list=[7,9]
    
    for theta, a_i, alpha_i, d_i in zip(joint_angles_radians, a, alpha, d):
        

        # apply edit gripper mdh for the control left down gripper and right down gripper
        if j==9:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        
        elif j==7:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        else:
            T_temp=create_transformation_matrix_mdh(theta, a_i, alpha_i, d_i)

        # gripper has different base coordinate so need new flip coordinate
        if flip_x_coordinate:
            if j not in gripper_index_list:
                T_temp=reflect_x_axis(T_temp) # for different base coordinate, the default base coordinate is - + - 
            else:
                # T_temp=create_transformation_matrix_mdh_gripper_reflect_x_coordinate(theta, a_i, alpha_i, d_i) # for different base coordinate, the default base coordinate is - + - since mdh gripper is different from mdh
                T_temp=reflect_x_axis_only(T_temp)
        # for gripper right 10,11, it is connect to 7 
        transformations.append(T_temp)

        j+=1
    # Calculate the final transformation from the base to the end-effector
    final_transformation = np.eye(4)
    final_transformations_list=[[]]*len(joint_angles_radians)

    p=0
    if add_gripper:

        
        for transformation in transformations:
            if p==9:
                gripper_move= final_transformations_list[6]
                final_transformation = np.dot(gripper_move, transformation)
                final_transformations_list[p]=final_transformation
            elif p==10:
                gripper_right_move= final_transformations_list[9]
                final_transformation = np.dot(gripper_right_move, transformation)
                final_transformations_list[p]=final_transformation
            else:
                final_transformation = np.dot(final_transformation, transformation)
                final_transformations_list[p]=final_transformation
            p+=1
    else:
        for transformation in transformations:
            final_transformation = np.dot(final_transformation, transformation)
            final_transformations_list[p]=final_transformation
            p+=1
    return transformations, final_transformations_list


# Function to calculate all transformation matrices and the final matrix
def calculate_transformations(i,joint_angles_degrees, a, alpha, d):
    """
    calulate transformation matrices based dh parameters

    Args:
    i: timestamp
    joint_angles_degrees: list of joint angles in degrees
    a: list of a parameters
    alpha: list of alpha parameters
    d: list of d parameters

    
    
    
    """
    joint_angles_radians = np.radians(joint_angles_degrees)
    transformations = []
    
    for theta, a_i, alpha_i, d_i in zip(joint_angles_radians, a, alpha, d):
        transformations.append(create_transformation_matrix(theta, a_i, alpha_i, d_i))
    
    # Calculate the final transformation from the base to the end-effector
    final_transformation = np.eye(4)
    for transformation in transformations:
        final_transformation = np.dot(final_transformation, transformation)
    
    return transformations, final_transformation

def bbox_corners(min_x, min_y, min_z, max_x, max_y, max_z):
    return [
        (min_x, min_y, min_z), # Front Bottom Left
        (max_x, min_y, min_z), # Front Bottom Right
        (min_x, max_y, min_z), # Back Bottom Left
        (max_x, max_y, min_z), # Back Bottom Right
        (min_x, min_y, max_z), # Front Top Left
        (max_x, min_y, max_z), # Front Top Right
        (min_x, max_y, max_z), # Back Top Left
        (max_x, max_y, max_z)  # Back Top Right
    ]




def inverse_affine_transformation(transforms):
    """
    Compute the inverse of an affine transformation matrix.

    Args:
    transformation_matrix: A 4x4 numpy array representing the transformation matrix.

    Returns:
    inverse_transformation_matrix: A 4x4 numpy array representing the inverse transformation matrix.
    """
    inv_transforms = []
    for transform in transforms:
        M = transform[:3, :3]  # Extract the rotation matrix M (top-left 3x3 submatrix)
        b = transform[:3, 3]   # Extract the translation vector b (top-right 3x1 subvector)
        
        M_inv = np.linalg.inv(M)  # Compute the inverse of M
        b_new = -M_inv @ b  # Compute -inv(M) * b
        
        # Construct the new transformation matrix inv(A)
        inv_A = np.zeros((4, 4))
        inv_A[:3, :3] = M_inv
        inv_A[:3, 3] = b_new
        inv_A[3, 3] = 1
        
        inv_transforms.append(inv_A)


    return inv_transforms


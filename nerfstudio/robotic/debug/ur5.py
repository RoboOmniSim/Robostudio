import torch
import open3d as o3d
import trimesh
import os
import numpy as np






import argparse
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.export_util.urdf_utils.urdf_config import *

# from nerfstudio.robotic.kinematic.gripper_utils import reflect_x_axis,reflect_y_axis,reflect_z_axis

from nerfstudio.robotic.kinematic.control_helper import *

from nerfstudio.robotic.export_util.urdf_utils.urdf_helper import *
from nerfstudio.robotic.config.raw_config import export_urdf_to_omnisim_config




def flip_axis(T,R_x):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    # R_x = np.array([
    #     [-1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    T_reflected = R_x @ T @ R_x
    return T_reflected



def calculate_transformations_mdh(state_position,joint_angles_degrees, a, alpha, d,flip):
    """
    calulate transformation matrices based dh parameters

    Args:
    i: timestamp
    joint_angles_degrees: list of joint angles in degrees
    a: list of a parameters
    alpha: list of alpha parameters
    d: list of d parameters

    
    
    
    """



    joint_angles_radians_raw = joint_angles_degrees
    joint_angle_deform= np.array(state_position)
    joint_angle_deform=np.round(joint_angle_deform, 3)



    joint_angles_radians=joint_angles_radians_raw+joint_angle_deform


    transformations = []

    j=0
    
    for theta, a_i, alpha_i, d_i in zip(joint_angles_radians, a, alpha, d):
        

       
        T_temp=create_transformation_matrix_mdh(theta, a_i, alpha_i, d_i)
        T_temp=flip_axis(T_temp,flip)
        transformations.append(T_temp)

        j+=1
    # Calculate the final transformation from the base to the end-effector
    final_transformation = np.eye(4)
    final_transformations_list=[[]]*len(joint_angles_radians)

    p=0

    for transformation in transformations:
            final_transformation = np.dot(final_transformation, transformation)
            final_transformations_list[p]=final_transformation
            p+=1
    return transformations, final_transformations_list





def test_rotation_matrix():

    # grasp dataset is the reference frame
    
    R_x = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])




    return R_x

def main():

    mesh_path = "./dataset/ur5/part"


    center_vector_gt=np.array([-0.12,0.7915,-0.64])
    scale_factor_pass= np.array([1.22,1.22,1.22])


    state_position=np.array([0,0,0,0,0,0])


    # find the initial position 
    R_x=np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    # invert
    # from urdf mdh parameters
    # a : [0,0,0.425,0.39225,0,0]
    # d : [0.089416,0,0,0.10915,0.09465,0.0823]
    # alpha : [0,1.57,0,0,1.57,-1.57]

    
    joint_angles_degrees =np.array([0,-1.57,-1.57,-1.57,1.57,0]) 
    a= np.array([0,0,0.425,0.39225,0,0])   
    d= np.array([0.089416,0,0,0.10915,0.09465,0.0823])
    alpha= np.array([0,1.57,0,0,1.57,-1.57])




    scale_a=np.array([1,1,1,1,1,1])/scale_factor_pass[2]  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
    scale_d=np.array([1,1,1,1,1,1])/scale_factor_pass[2] #  s1 is x axis for d1 ,s4 and s6  is y axis 



    scale_d[3]=1/scale_factor_pass[1]
    scale_d[5]=1/scale_factor_pass[1]

    a=a*scale_a
    d=d*scale_d
    # forward

    transformations, final_transformations_list_0=calculate_transformations_mdh(state_position=state_position,joint_angles_degrees=joint_angles_degrees,a=a,alpha=alpha,d=d,flip=R_x)



    state_position=np.array([0,0.5,0,0,0,0])  # edit this for deformation


    transformations, final_transformations_list=calculate_transformations_mdh(state_position=state_position,joint_angles_degrees=joint_angles_degrees,a=a,alpha=alpha,d=d,flip=R_x)




    inverse_transformation= inverse_affine_transformation(final_transformations_list_0)


    # 


    for i in range(len(final_transformations_list_0)):
        rotation_inv=inverse_transformation[i][:3,:3]
        translation_inv=inverse_transformation[i][:3,3]

        rotation=final_transformations_list[i][:3,:3]
        translation=final_transformations_list[i][:3,3]

        raw_xyz=raw_xyz-center_vector_gt
        deform_point=  np.array(raw_xyz @ rotation_inv.T+ translation_inv )

        forward_point=  np.array(deform_point @ rotation.T+ translation )

        select_xyz=forward_point+center_vector_gt
        rotation_splat=rotation@rotation_inv

    return 0
















if __name__=="__main__":
    main()


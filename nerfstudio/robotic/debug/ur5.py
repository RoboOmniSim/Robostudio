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


def rotation_matrix(roll, pitch, yaw):
    """
    Generate a transformation matrix from roll, pitch, and yaw angles.
    :param roll: Rotation around the X-axis (in radians)
    :param pitch: Rotation around the Y-axis (in radians)
    :param yaw: Rotation around the Z-axis (in radians)
    :return: 4x4 homogeneous transformation matrix
    """
    # Rotation around the X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation around the Y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around the Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Add translation (if necessary, here we use identity)
    T = np.identity(4)
    T[:3, :3] = R

    return T




def load_ply_files(input_path):
    points_list = []
    color_list = []
    normal_list = []
    
    # Iterate through all files in the directory

    path_list=os.listdir(input_path)
    path_list.sort()
    for file_name in path_list:
        if file_name.endswith(".ply"):
            # Load the point cloud
            file_path = os.path.join(input_path, file_name)
            point_cloud = o3d.io.read_point_cloud(file_path)
            
            # Get points (Nx3 numpy array)
            points = np.asarray(point_cloud.points)
            points_list.append(points)
            
            # Get colors (Nx3 numpy array)
            if point_cloud.has_colors():
                colors = np.asarray(point_cloud.colors)
                color_list.append(colors)
            else:
                # If no color information is available, append None or zeros
                color_list.append(np.zeros_like(points))
            
            # Get normals (Nx3 numpy array)
            if point_cloud.has_normals():
                normals = np.asarray(point_cloud.normals)
                normal_list.append(normals)
            else:
                # If no normal information is available, compute normals or append None
                point_cloud.estimate_normals()
                normals = np.asarray(point_cloud.normals)
                normal_list.append(normals)
    
    return points_list, color_list, normal_list


def apply_rotation_on_point_cloud(points_list, rotation_matrix):
    """
    Apply rotation to a point cloud using a rotation matrix.
    
    Args:
        points (numpy.ndarray): Nx3 array of points.
        rotation_matrix (numpy.ndarray): 4*4 transformation matrix.
    
    Returns:
        numpy.ndarray: Rotated Nx3 array of points.
    """

    for i in range(len(points_list)):
        points_list[i] = points_list[i] @ rotation_matrix[:3,:3].T
    return points_list






def save_ply_file(output_path, points, colors=None, normals=None):
    """
    Save a point cloud to a PLY file.
    
    Args:
        output_path (str): Path to save the PLY file.
        points (numpy.ndarray): Nx3 array of points.
        colors (numpy.ndarray, optional): Nx3 array of colors. Defaults to None.
        normals (numpy.ndarray, optional): Nx3 array of normals. Defaults to None.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    
    o3d.io.write_point_cloud(output_path, point_cloud)










def main():

    ply_path = "./dataset/ur5/part"
    ply_save_path= "./dataset/ur5/part/reorient"
    deform_ply_save_path= "./dataset/ur5/part/deform"


    if not os.path.exists(ply_save_path):
        os.makedirs(ply_save_path)

    if not os.path.exists(deform_ply_save_path):
        os.makedirs(deform_ply_save_path)
        
    points_list, color_list, normal_list=load_ply_files(ply_path)

    # first step, test and fix chiral

    # edit the roll pitch yaw here i just give a start, but u need to fix it by change the value 

    # compare the reoriented and grasp 
    roll = np.pi
    pitch = np.pi  
    yaw = 0

    # find the initial position 


    R_x=rotation_matrix(roll=roll, pitch=pitch, yaw=yaw)
    
    print("R_x",R_x)


    point_list_reori=apply_rotation_on_point_cloud(points_list, R_x)
    

    for i in range(len(points_list)):
        save_ply_file(os.path.join(ply_save_path, f"reoriented_{i}.ply"), point_list_reori[i], color_list[i], normal_list[i])


    # this you will see the reorientation result


    center_vector_gt=np.array([-0.12,0.7915,-0.64]) # if there is same gap when you edit other value and has no difference in the center, you can change this value
    scale_factor_pass= np.array([1.22,1.22,1.22]) # this is editable


    state_position=np.array([0,0,0,0,0,0])





    # invert
    # from urdf mdh parameters
    # a : [0,0,0.425,0.39225,0,0]
    # d : [0.089416,0,0,0.10915,0.09465,0.0823]
    # alpha : [0,1.57,0,0,1.57,-1.57]


    joint_angles_degrees =np.array([0,-1.57,-1.57,-1.57,1.57,0]) # should be good
    a= np.array([0,0,0.425,0.39225,0,0])   # minor edit should be good 
    d= np.array([0.089416,0,0,0.10915,0.09465,0.0823])
    alpha= np.array([0,1.57,0,0,1.57,-1.57]) # if you have any value changed but has weird rotation, change this




    scale_a=np.array([1,1,1,1,1,1])/scale_factor_pass[2]  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
    scale_d=np.array([1,1,1,1,1,1])/scale_factor_pass[2] #  s1 is x axis for d1 ,s4 and s6  is y axis 



    scale_d[3]=1/scale_factor_pass[1] # i am not sure about this index 3 and 5, you can refer to the standard dh table, we want this to be the y axis change
    scale_d[5]=1/scale_factor_pass[1]

    a=a*scale_a
    d=d*scale_d
    # forward

    #map back to center
    transformations, final_transformations_list_0=calculate_transformations_mdh(state_position=state_position,joint_angles_degrees=joint_angles_degrees,a=a,alpha=alpha,d=d,flip=R_x)



    state_position=np.array([0,0.5,0,0,0,0])  # edit this for deformation

    # change for new trajectory
    transformations, final_transformations_list=calculate_transformations_mdh(state_position=state_position,joint_angles_degrees=joint_angles_degrees,a=a,alpha=alpha,d=d,flip=R_x)




    inverse_transformation= inverse_affine_transformation(final_transformations_list_0)


    # 

    save_point_list=[]
    for i in range(len(final_transformations_list_0)):
        rotation_inv=inverse_transformation[i][:3,:3]
        translation_inv=inverse_transformation[i][:3,3]

        rotation=final_transformations_list[i][:3,:3]
        translation=final_transformations_list[i][:3,3]

        raw_xyz=point_list_reori[i]
        raw_xyz=raw_xyz-center_vector_gt
        deform_point=  np.array(raw_xyz @ rotation_inv.T+ translation_inv )

        forward_point=  np.array(deform_point @ rotation.T+ translation )

        select_xyz=forward_point+center_vector_gt
        rotation_splat=rotation@rotation_inv


        save_point_list.append(select_xyz)
    
    for i in range(len(save_point_list)):
        save_ply_file(os.path.join(deform_ply_save_path, f"deformed_{i}.ply"), save_point_list[i], color_list[i], normal_list[i])


    return 0
















if __name__=="__main__":
    main()


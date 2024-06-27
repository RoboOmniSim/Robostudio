import numpy as np
import trimesh
import open3d as o3d








from nerfstudio.robotic.physics_engine.python.collision_detection import collision_detection



from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.kinematic.gripper_utils import *

from scipy.spatial.transform import Rotation as R



def rotation_matrix(axis, angle):
    """
    Compute the rotation matrix for a given axis and angle.
    
    Parameters:
    axis (str): 'x', 'y', or 'z' axis
    angle (float): Angle in radians
    
    Returns:
    np.ndarray: 3x3 rotation matrix
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', 'z'.")





def interact_with_gripper(euler_gripper,dt):
    """
    
    The interaction between the gripper and the object after the grasp start and before the grasp success
    
    """
    transformation=np.eye(4)
    
    manipulate_vector=0.015
    interpolate_value=manipulate_vector*dt/10
    # print('dt',dt)
    # print('interpolate_value',interpolate_value)
    translation=transformation[:3,3]
    translation[2]=translation[2]+interpolate_value # also need interpolation
    translation[0]=translation[0]+0
    euler=euler_gripper
    axis='y'
    rotation = rotation_matrix(axis,euler_gripper)
    return rotation, translation

def obtain_relative_transformation(T_0,T_1):
        return np.dot(np.linalg.inv(T_0),T_1)

def get_object_tracjetory_from_simulation(mark_id,mark_id_offset,start_time_stamp,time_stamp,final_transformations_list,
                                          movement_angle_state,joint_angles_degrees, a, alpha, d,gripper_control,
                                          joint_angles_degrees_gripper, a_gripper, alpha_gripper, d_gripper,
                                          flip_x_coordinate=False,add_grasp_control=False):


    individual_transformations_t_minus1, final_transformations_list_t_minus1=gripper_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,gripper_control,
                                                                                joint_angles_degrees_gripper, a_gripper, alpha_gripper, d_gripper,
                                                                                i=start_time_stamp,flip_x_coordinate=flip_x_coordinate,add_grasp_control=add_grasp_control)

       # x
    dynamic_scale_x=np.ones(3)
    dynamic_scale_z=np.ones(3)

    # this is the manual linear regressin parameter for object movement
    if time_stamp < 265 :
        dynamic_scale_x[0]=0.7
        dynamic_scale_x[1]=0.5 # should avoid rotation of arm?
        dynamic_scale_x[2]=0.99
        dynamic_scale_z[0]=0.85
        dynamic_scale_z[1]=0.01
        dynamic_scale_z[2]=0.72
    elif 290> time_stamp>=265:
        dynamic_scale_x[0]=0.8
        dynamic_scale_x[1]=0.4
        dynamic_scale_x[2]=0.99
        dynamic_scale_z[0]=0.85
        dynamic_scale_z[1]=0.01
        dynamic_scale_z[2]=0.84
    elif 300>= time_stamp>=290:
        dynamic_scale_x[0]=0.8
        dynamic_scale_x[1]=0.25
        dynamic_scale_x[2]=0.99
        dynamic_scale_z[0]=0.85
        dynamic_scale_z[1]=0.1
        dynamic_scale_z[2]=0.83
    else:
        dynamic_scale_x[0]=0.95
        dynamic_scale_x[1]=0.01
        dynamic_scale_x[2]=0.99
        dynamic_scale_z[0]=0.85
        dynamic_scale_z[1]=0.01
        dynamic_scale_z[2]=0.78
    relative_trans_offset=obtain_relative_transformation(final_transformations_list[mark_id_offset],final_transformations_list_t_minus1[mark_id_offset])
    # offset_vector=np.array((0,0,0))
    offset_vector=relative_trans_offset[:3,3]
    offset_rotation=relative_trans_offset[:3,:3]
    offset_vector[0]=offset_vector[0]*dynamic_scale_x[0]
    offset_vector[1]=offset_vector[1]*dynamic_scale_x[1]
    offset_vector[2]=offset_vector[2]*dynamic_scale_z[2]

    relative_trans=obtain_relative_transformation(final_transformations_list[mark_id],final_transformations_list_t_minus1[mark_id])

    # rotation_rela=relative_trans[:3,:3]
    rotation_rela=np.eye(3)
    translation_rela=relative_trans[:3,3]
    # translation_rela[2]=0
    translation_rela[0]= translation_rela[0]*dynamic_scale_z[0]
    translation_rela[1]= translation_rela[1]*dynamic_scale_z[1]
    translation_rela[2]= translation_rela[2]*dynamic_scale_z[2]


    return rotation_rela,translation_rela,offset_vector,offset_rotation
def gripper_self_detection(gripper_left_up,gripper_right_up,gripper_deform):
    """
    Args:
        gripper_left_up (np.ndarray): left finger mesh.
        gripper_right_up (np.ndarray): right finger mesh.
        gripper_deform (int): the number of deformation for the gripper.
    Returns:
        time_list (list): the time list for the gripper detection, True is collision, False is no collision.
    
    
    """
    time_list=[[]]*gripper_deform
    for i in range(0,gripper_deform):
        inverse=np.eye(4)
        transform1=transformation_matrix(gripper_deform) # left finger
        transform2=transformation_matrix(gripper_deform)    # right finger
        result=collision_detection(gripper_left_up,gripper_right_up,inverse,transform1,transform2)
        time_list[i]=result
    return time_list


def gripper_object_detection(gripper_left_up,gripper_right_up,object_mesh,inverse,transform1,transform2):
    """

    This function is used to detect the collision between the gripper and the object.

    The result_1 is the collision detection between the left finger and the object.
    The result_2 is the collision detection between the right finger and the object.

    if this two result did not have collision at same time, then the object is moved by the gripper.

    Args:
        gripper_left_up (np.ndarray): left finger mesh.
        gripper_right_up (np.ndarray): right finger mesh.
        object_mesh (np.ndarray): object mesh.
        inverse: the recenter matrix
        transform (np.ndarray): Transformation matrix to apply to object.
    Returns:
        time_list (list): the time list for the gripper detection, True is collision, False is no collision.
    
    
    
    
    """

    time_list_1=[[]]*gripper_deform
    time_list_2=[[]]*gripper_deform
    for i in range(0,gripper_deform):
        identity=np.eye(4)
        inverse=np.eye(4)
        transform1=transformation_matrix(gripper_deform) # left finger
        transform2=transformation_matrix(gripper_deform)    # right finger
        result_1=collision_detection(gripper_left_up,object_mesh,inverse,transform1,identity)
        time_list_1[i]=result_1
        result_2=collision_detection(gripper_right_up,object_mesh,inverse,transform2,identity)
        time_list_2[i]=result_2

    return time_list_1, time_list_2


def object_gripper_move(collision_time_1,collision_time_2,object):
    """
    Args:
        collision_time_1 (list): the time list for the gripper detection, True is collision, False is no collision.
        collision_time_2 (list): the time list for the gripper detection, True is collision, False is no collision.
        object (np.ndarray): object mesh.
    Returns:
        object (np.ndarray): object mesh after moving by the gripper.
        relative_transformation (np.ndarray): the relative transformation between the object in the start and  end.



    This function is used to move the object by the gripper.
    THe general idea is the gripper touch object did not at the same time, so we can move the object. to the target optimal position.
    
    
    """
    


    for i in range(0,gripper_deform):
        if collision_time_1[i]==False and collision_time_2[i]==False:
            # move the object
            # get the relative transformation between the object in the start and end
            relative_transformation=np.eye(4)
            return object, relative_transformation
    return object, np.eye(4)



def gripper_grasp_movement(gripper_left_up,gripper_right_up,object_mesh,inverse,transform1,transform2):
    """
    Args:
        gripper_left_up (np.ndarray): left finger mesh.
        gripper_right_up (np.ndarray): right finger mesh.
        object_mesh (np.ndarray): object mesh.
        inverse: the recenter matrix
        transform (np.ndarray): Transformation matrix to apply to object.
    Returns:
        object (np.ndarray): object mesh after moving by the gripper.
        relative_transformation (np.ndarray): the relative transformation between the object in the start and  end.
    
    
    This function is used to move the rigid object with gripper after the grasp success.
    
    """
    collision_time_1, collision_time_2=gripper_object_detection(gripper_left_up,gripper_right_up,object_mesh,inverse,transform1,transform2)
    object, relative_transformation=object_gripper_move(collision_time_1,collision_time_2,object_mesh)
    return  object, relative_transformation





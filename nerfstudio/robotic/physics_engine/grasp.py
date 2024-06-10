import numpy as np
import trimesh
import open3d as o3d








from nerfstudio.robotic.physics_engine.collision_detection import collision_detection



from nerfstudio.robotic.kinematic.uniform_kinematic import *


from scipy.spatial.transform import Rotation as R







# we need to use the mesh box to keep the stop distance of gripper 






# and make the object as the same recenter and transformation of the gripper 



# collision detection between the gripper final part convex mesh and object


# find the self closed angle and the angle range for right and left finger


# test the kinematic first, and then set up same angle range and pass to collistion detection for visual check




# then setup control signal



def interact_with_gripper(gripper_left_up,gripper_right_up,object_mesh,dt):
    """
    
    The interaction between the gripper and the object after the grasp start and before the grasp success
    
    """

    return rotation, translation


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





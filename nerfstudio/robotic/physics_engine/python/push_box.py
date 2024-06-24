# cite from Jatavallabhula and Macklin et al., "gradSim: Differentiable simulation for system identification and visuomotor control", ICLR 2021.



import numpy as np
import trimesh
import open3d as o3d








from nerfstudio.robotic.physics_engine.python.collision_detection import collision_detection



from nerfstudio.robotic.kinematic.uniform_kinematic import *


from scipy.spatial.transform import Rotation as R

import pypose as pp
from nerfstudio.robotic.physics_engine.python.backward import *

def compute_torque(force, application_point):
    r"""Compute the torque due to a force applied at a specific point.

    Args:
        force (torch.Tensor): Force vector (shape: :math:`(3)`).
        application_point (torch.Tensor): Application point of the force (shape: :math:`(3)`).

    Returns:
        (torch.Tensor): Torque vector (shape: :math:`(3)`).
    """
    return np.cross(application_point, force)

def compute_moment_of_inertia(mass, dimensions):
    # Assuming the box is a rectangular prism with dimensions [width, height, depth]
    width, height, depth = dimensions
    I_xx = (1/12) * mass * (height**2 + depth**2)
    I_yy = (1/12) * mass * (width**2 + depth**2)
    I_zz = (1/12) * mass * (width**2 + height**2)
    return np.array([I_xx, I_yy, I_zz])

def compute_angular_acceleration(torque, moment_of_inertia):
    return torque / moment_of_inertia

def integrate_angular_acceleration(angular_acceleration, dt):
    return angular_acceleration * dt



def compute_linear_velocity(angular_velocity, position_vector):
    return np.cross(angular_velocity, position_vector)


def leverage_simulation(mesh,center_of_mass,fulcrum,linear_velocity,angular_velocity,dt):

    # the start center is the center of obj bbox




    # the end center is the center when object is laying on the ground (ie,the bottom of orignial bbox + the center y axis)

    # center_of_mass = np.array([0.0, 0.0, 0.0])  # Initial center of mass
    # fulcrum = np.array([0.5, 0.5, 0.0])  # Fulcrum position
    # linear_velocity = np.array([1.0, 0.0, 0.0])  # Linear velocity
    # angular_velocity = np.array([0.0, 0.0, 1.0])  # Angular velocity (rotation around z-axis)
      # Time step

    # Compute global transformation
    new_rotation_matrix, translation_vector = compute_global_transformation(center_of_mass, fulcrum, linear_velocity, angular_velocity, dt)

    return new_rotation_matrix, translation_vector




def compute_angular_velocity(xyz_0,xyz_optimized,delta_t):





    # angular velocity very time consuming so replace it in future 



    # Example vertex positions before and after movement
    # Each row represents a vertex. Column are the x, y, z coordinates
    vertices_initial = xyz_0
    vertices_final = xyz_optimized

    # Using SVD to find the optimal rotation matrix
    U, S, Vt = np.linalg.svd(vertices_initial.T @ vertices_final)
    rotation_matrix = U @ Vt

    # Ensure a proper rotation matrix (handling possible reflection)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = U @ Vt

    # Convert rotation matrix to axis-angle
    rot = R.from_matrix(rotation_matrix)
    angle = rot.magnitude()
    axis = rot.as_rotvec() / angle

    # Time interval (example: 1 second)
    delta_t = 1.0

    # Angular velocity
    angular_velocity = axis * angle / delta_t


    return angular_velocity



def update_position(center_of_mass, linear_velocity, dt):
    # Calculate new position based on linear velocity
    return center_of_mass + linear_velocity * dt

def update_orientation(rotation_matrix, angular_velocity, dt):
    # Calculate angular displacement
    angular_displacement = angular_velocity * dt
    # Convert angular displacement to a rotation matrix
    rotation = R.from_rotvec(angular_displacement).as_matrix()
    # Update the current rotation matrix
    new_rotation_matrix = rotation @ rotation_matrix
    return new_rotation_matrix

def compute_global_transformation(center_of_mass, fulcrum, linear_velocity, angular_velocity, dt):
    # Compute new position of the center of mass
    new_center_of_mass = update_position(center_of_mass, linear_velocity, dt)
    
    # Initial rotation matrix (identity if no initial rotation)
    initial_rotation_matrix = np.eye(3)
    
    # Compute new orientation
    new_rotation_matrix = update_orientation(initial_rotation_matrix, angular_velocity, dt)
    
    # Calculate the translation vector from fulcrum to the new center of mass
    translation_vector = new_center_of_mass - fulcrum
    
    return new_rotation_matrix, translation_vector

def concat_transformation(rotation_matrix, translation_vector):
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix



def find_lower_plane_center(corners):
    """
    Find the center of the lower plane of a bounding box.
    
    :param corners: List of 8 corners of the bounding box. Each corner is a tuple (x, y, z).
    :return: Tuple representing the center of the lower plane (x, y, z).
    """
    # Sort the corners based on the z-coordinate
    sorted_corners = sorted(corners, key=lambda corner: corner[2])
    
    # The first four corners (after sorting) will be the lower plane corners
    lower_plane_corners = sorted_corners[:4]
    
    # Calculate the center of the lower plane
    lower_plane_center = np.mean(lower_plane_corners, axis=0)
    
    return lower_plane_center



def example_push(dt):
       # Box properties
    mass = 10  # mass of the box
    # dimensions = [1.0, 1.0, 1.0]  # dimensions of the box (width, height, depth)
    center_of_mass=np.array([0,0,0])
    # for push case:
    # gripper_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/gripper_convex.obj' # id 7
    object_mesh_path='./Robostudio/dataset/push_box/part/object/box_convex.obj'  # id 8 
    # ground_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/table_convex.obj'

    # gripper_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/gripper_convex.obj'
    # object_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/box_convex.obj'
    # ground_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/table_convex.obj'

    mesh_object = trimesh.load(object_mesh_path)
    # mesh_ground = trimesh.load(ground_mesh_path)
    # gripper_mesh = trimesh.load(gripper_mesh_path)

    bb=mesh_object.bounding_box_oriented

    corners=trimesh.bounds.corners(mesh_object.bounding_box_oriented.bounds)

    recenter_vector=find_lower_plane_center(corners)

    

    # mesh_object.apply_translation(-recenter_vector)

    # mesh_object.export('/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/mesh_object_recenter.obj')

    dimensions=bb.extents

    bb_recentered=mesh_object.bounding_box_oriented
    center_of_mass=bb_recentered.centroid
    # Compute the moment of inertia
    moment_of_inertia = compute_moment_of_inertia(mass, dimensions)
    force=np.array([0.1,0,0])


    simulation_position=np.array([0.0, 0.0, 0.013])
    # Given torque applied (for example purposes)
    force_position = center_of_mass+ simulation_position # torque vector
    torque=compute_torque(force, force_position)  # torque vector
    # Time step for integration


    # Compute angular acceleration
    angular_acceleration = compute_angular_acceleration(torque, moment_of_inertia)

    # Integrate angular acceleration to get angular velocity
    angular_velocity = integrate_angular_acceleration(angular_acceleration, dt)

    fulfrum_position=np.array([0,0,0]) # the position of the fulcrum
    # force_position=np.array([1,0,0])
    

    # Position vector of the point where you want to compute the linear velocity
    position_vector = force_position- fulfrum_position

    linear_velocity = compute_linear_velocity(angular_velocity, position_vector)
    # this should be pre_timestamp 4*4 transformation matrix that perform the 



    # experiment_type='push_bag' # novelpose or push_bag
    # # gripper also need to contact with object to determine the start time

    # # experiment_type='novelpose' # novelpose or push_bag
    # experiment_type='push_bag' # novelpose or push_bag


    # if experiment_type=='novelpose':

    #         output_file = "/home/lou/gs/nerfstudio/transformation_novelpose/joint_states_data.txt" # novel posedata
    #         static_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/splat.ply' # for novelpose
    # elif experiment_type=='push_bag':
            
    #         output_file = "/home/lou/gs/nerfstudio/transformation_group1/joint_states_data_push.txt" # push bbox data
    #         static_path= '/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/splat.ply'
    # elif experiment_type=='grasp':  # grasp data for the gripper only 
    #         output_file=None
    #         static_path=None
    # elif experiment_type=='grasp_object':  # grasp data for the gripper and object
    #         output_file="/home/lou/gs/nerfstudio/transformation_0416_object_grasp/joint_states_data_0416.txt"
    #         static_path="/home/lou/Downloads/gripper_movement/gripper_part_asset/splat.ply"
    # else:
    #         print('experiment type not found')
    #         raise ValueError('experiment type not found')

    # movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file,experiment_type)

    # time_stamp= np.linspace(0,850,850)
    # gripper_move_list=[0]*len(time_stamp)
    # for i in range(len(time_stamp)):
    #     individual_transformations, final_transformations_list = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=i)
    #     gripper_move_list[i]=final_transformations_list[5]
    #     # apply transformation:



    #     # test inverse

    # inverse_transformation=inverse_affine_transformation(final_transformations_list_0)

    # gripper_inverse=inverse_transformation[5]
    # detection_signal=[False]*len(gripper_move_list)

    # for i in range(len(detection_signal)):
    #     # determine end time based on the 
    #     detect=collision_detection(mesh_object,gripper_mesh,gripper_inverse,gripper_move_list[i])   
    #     detection_signal[i]=detect

    # start_moment=0

    # end_moment=0
    # i=0
    
    # for time in detection_signal:
    #     if time==True:
    #         start_moment=i
    #     else:
    #         i+=1

    
    # get the start and end center of the object

    new_rotation_matrix, translation_vector=leverage_simulation(mesh_object,center_of_mass,fulfrum_position,linear_velocity,angular_velocity,dt)
    
    trans=concat_transformation(new_rotation_matrix, translation_vector)



    # mesh_object.apply_transform(trans)

    eye_rotate=np.eye(3)

    back=concat_transformation(eye_rotate, -simulation_position)
    # mesh_object.apply_transform(back)

    recenter_back_vector=recenter_vector

    # mesh_object.apply_translation(recenter_back_vector)
    # mesh_object.apply_translation(translation_vector)

    # save mesh
    # mesh_object.export('/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/mesh_object_dt_10_t_05.obj')
    # when object center reach the end position is the end time
    # end_time



    # only deform object from start time to end time

    return recenter_vector,new_rotation_matrix, translation_vector,simulation_position
# add_collision here hhh 


def add_simulation_control(mesh_object,center_of_mass,fulcrum,linear_velocity,angular_velocity,dt):


    return 0



def example_push_backward(dt):
       # Box properties
    mass = 10  # mass of the box
    # dimensions = [1.0, 1.0, 1.0]  # dimensions of the box (width, height, depth)
    center_of_mass=np.array([0,0,0])
    # for push case:
    # gripper_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/gripper_convex.obj' # id 7
    object_mesh_path='./Robostudio/dataset/push_box/part/object/box_convex.obj'  # id 8 
    # ground_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/table_convex.obj'

    # gripper_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/gripper_convex.obj'
    # object_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/box_convex.obj'
    # ground_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/table_convex.obj'

    mesh_object = trimesh.load(object_mesh_path)
    # mesh_ground = trimesh.load(ground_mesh_path)
    # gripper_mesh = trimesh.load(gripper_mesh_path)

    bb=mesh_object.bounding_box_oriented

    corners=trimesh.bounds.corners(mesh_object.bounding_box_oriented.bounds)

    recenter_vector=find_lower_plane_center(corners)

    

    # mesh_object.apply_translation(-recenter_vector)

    # mesh_object.export('/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/mesh_object_recenter.obj')

    dimensions=bb.extents

    bb_recentered=mesh_object.bounding_box_oriented
    center_of_mass=bb_recentered.centroid
    # Compute the moment of inertia
    moment_of_inertia = compute_moment_of_inertia(mass, dimensions)
    force=np.array([0.1,0,0])


    simulation_position=np.array([0.0, 0.0, 0.013])
    # Given torque applied (for example purposes)
    force_position = center_of_mass+ simulation_position # torque vector
    torque=compute_torque(force, force_position)  # torque vector
    # Time step for integration


    # Compute angular acceleration
    angular_acceleration = compute_angular_acceleration(torque, moment_of_inertia)

    # Integrate angular acceleration to get angular velocity
    angular_velocity = integrate_angular_acceleration(angular_acceleration, dt)

    fulfrum_position=np.array([0,0,0]) # the position of the fulcrum
    # force_position=np.array([1,0,0])
    

    # Position vector of the point where you want to compute the linear velocity
    position_vector = force_position- fulfrum_position

    linear_velocity = compute_linear_velocity(angular_velocity, position_vector)
    # this should be pre_timestamp 4*4 transformation matrix that perform the 



    # experiment_type='push_bag' # novelpose or push_bag
    # # gripper also need to contact with object to determine the start time

    # # experiment_type='novelpose' # novelpose or push_bag
    # experiment_type='push_bag' # novelpose or push_bag


    # if experiment_type=='novelpose':

    #         output_file = "/home/lou/gs/nerfstudio/transformation_novelpose/joint_states_data.txt" # novel posedata
    #         static_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/splat.ply' # for novelpose
    # elif experiment_type=='push_bag':
            
    #         output_file = "/home/lou/gs/nerfstudio/transformation_group1/joint_states_data_push.txt" # push bbox data
    #         static_path= '/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/splat.ply'
    # elif experiment_type=='grasp':  # grasp data for the gripper only 
    #         output_file=None
    #         static_path=None
    # elif experiment_type=='grasp_object':  # grasp data for the gripper and object
    #         output_file="/home/lou/gs/nerfstudio/transformation_0416_object_grasp/joint_states_data_0416.txt"
    #         static_path="/home/lou/Downloads/gripper_movement/gripper_part_asset/splat.ply"
    # else:
    #         print('experiment type not found')
    #         raise ValueError('experiment type not found')

    # movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file,experiment_type)

    # time_stamp= np.linspace(0,850,850)
    # gripper_move_list=[0]*len(time_stamp)
    # for i in range(len(time_stamp)):
    #     individual_transformations, final_transformations_list = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=i)
    #     gripper_move_list[i]=final_transformations_list[5]
    #     # apply transformation:



    #     # test inverse

    # inverse_transformation=inverse_affine_transformation(final_transformations_list_0)

    # gripper_inverse=inverse_transformation[5]
    # detection_signal=[False]*len(gripper_move_list)

    # for i in range(len(detection_signal)):
    #     # determine end time based on the 
    #     detect=collision_detection(mesh_object,gripper_mesh,gripper_inverse,gripper_move_list[i])   
    #     detection_signal[i]=detect

    # start_moment=0

    # end_moment=0
    # i=0
    
    # for time in detection_signal:
    #     if time==True:
    #         start_moment=i
    #     else:
    #         i+=1

    
    # get the start and end center of the object

    new_rotation_matrix, translation_vector=leverage_simulation(mesh_object,center_of_mass,fulfrum_position,linear_velocity,angular_velocity,dt)
    
    trans=concat_transformation(new_rotation_matrix, translation_vector)

    eye_rotate=np.eye(3)

    back=concat_transformation(eye_rotate, -simulation_position)





    pts_optimized,velocity,optimized_pose=backward_rigid(xyz,trans,back,uv,depth,projection_matrix,view_mat,dt)
    # mesh_object.apply_transform(trans)


    # mesh_object.apply_transform(back)

    recenter_back_vector=recenter_vector

    # mesh_object.apply_translation(recenter_back_vector)
    # mesh_object.apply_translation(translation_vector)

    # save mesh
    # mesh_object.export('/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/mesh_object_dt_10_t_05.obj')
    # when object center reach the end position is the end time
    # end_time



    # only deform object from start time to end time

    return recenter_vector,new_rotation_matrix, translation_vector,simulation_position



# just for debug

if __name__ == "__main__":
    
    
    # scale of contraction
    scale_factor=1.0
    
    # Box properties
    mass = 10  # mass of the box
    # dimensions = [1.0, 1.0, 1.0]  # dimensions of the box (width, height, depth)
    center_of_mass=np.array([0,0,0])
    # for push case:
    # gripper_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/gripper_convex.obj' # id 7
    object_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/box_convex.obj'  # id 8 
    # ground_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/table_convex.obj'

    # gripper_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/gripper_convex.obj'
    # object_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/box_convex.obj'
    # ground_mesh_path='/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object_mesh/table_convex.obj'

    mesh_object = trimesh.load(object_mesh_path)
    # mesh_ground = trimesh.load(ground_mesh_path)
    # gripper_mesh = trimesh.load(gripper_mesh_path)

    bb=mesh_object.bounding_box_oriented

    corners=trimesh.bounds.corners(mesh_object.bounding_box_oriented.bounds)

    recenter_vector=find_lower_plane_center(corners)

    mesh_object.apply_translation(-recenter_vector)

    mesh_object.export('/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/mesh_object_recenter.obj')

    dimensions=bb.extents

    bb_recentered=mesh_object.bounding_box_oriented
    center_of_mass=bb_recentered.centroid
    # Compute the moment of inertia
    moment_of_inertia = compute_moment_of_inertia(mass, dimensions)
    force=np.array([0.1,0,0])

    # replace by the final pose of gripper and its intersection with the object
    simulation_position=np.array([0.0, 0.0, 0.025])
    # Given torque applied (for example purposes)
    force_position = center_of_mass+ simulation_position # torque vector
    torque=compute_torque(force, force_position)  # torque vector
    # Time step for integration
    dt = 0.8  # time step 0-1

    # Compute angular acceleration
    angular_acceleration = compute_angular_acceleration(torque, moment_of_inertia)

    # Integrate angular acceleration to get angular velocity
    angular_velocity = integrate_angular_acceleration(angular_acceleration, dt)

    fulfrum_position=np.array([0,0,0]) # the position of the fulcrum
    # fulfrum_position=recenter_vector # the position of the fulcrum
    # force_position=np.array([1,0,0])
    

    # Position vector of the point where you want to compute the linear velocity
    position_vector = force_position- fulfrum_position

    linear_velocity = compute_linear_velocity(angular_velocity, position_vector)
    # this should be pre_timestamp 4*4 transformation matrix that perform the 



    # experiment_type='push_bag' # novelpose or push_bag
    # # gripper also need to contact with object to determine the start time

    # # experiment_type='novelpose' # novelpose or push_bag
    # experiment_type='push_bag' # novelpose or push_bag


    # if experiment_type=='novelpose':

    #         output_file = "/home/lou/gs/nerfstudio/transformation_novelpose/joint_states_data.txt" # novel posedata
    #         static_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/splat.ply' # for novelpose
    # elif experiment_type=='push_bag':
            
    #         output_file = "/home/lou/gs/nerfstudio/transformation_group1/joint_states_data_push.txt" # push bbox data
    #         static_path= '/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/splat.ply'
    # elif experiment_type=='grasp':  # grasp data for the gripper only 
    #         output_file=None
    #         static_path=None
    # elif experiment_type=='grasp_object':  # grasp data for the gripper and object
    #         output_file="/home/lou/gs/nerfstudio/transformation_0416_object_grasp/joint_states_data_0416.txt"
    #         static_path="/home/lou/Downloads/gripper_movement/gripper_part_asset/splat.ply"
    # else:
    #         print('experiment type not found')
    #         raise ValueError('experiment type not found')

    # movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file,experiment_type)

    # time_stamp= np.linspace(0,850,850)
    # gripper_move_list=[0]*len(time_stamp)
    # for i in range(len(time_stamp)):
    #     individual_transformations, final_transformations_list = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=i)
    #     gripper_move_list[i]=final_transformations_list[5]
    #     # apply transformation:



    #     # test inverse

    # inverse_transformation=inverse_affine_transformation(final_transformations_list_0)

    # gripper_inverse=inverse_transformation[5]
    # detection_signal=[False]*len(gripper_move_list)

    # for i in range(len(detection_signal)):
    #     # determine end time based on the 
    #     detect=collision_detection(mesh_object,gripper_mesh,gripper_inverse,gripper_move_list[i])   
    #     detection_signal[i]=detect

    # start_moment=0

    # end_moment=0
    # i=0
    
    # for time in detection_signal:
    #     if time==True:
    #         start_moment=i
    #     else:
    #         i+=1

    
    # get the start and end center of the object

    new_rotation_matrix, translation_vector=leverage_simulation(mesh_object,center_of_mass,fulfrum_position,linear_velocity,angular_velocity,dt)
    
    trans=concat_transformation(new_rotation_matrix, translation_vector)

    mesh_object.apply_transform(trans)

    eye_rotate=np.eye(3)

    back=concat_transformation(eye_rotate, -simulation_position)
    mesh_object.apply_transform(back)

    recenter_back_vector=recenter_vector

    mesh_object.apply_translation(recenter_back_vector)
    # mesh_object.apply_translation(translation_vector)

    # save mesh
    mesh_object.export('/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/object_convex/mesh_object_dt_10_t_05.obj')
    # when object center reach the end position is the end time
    # end_time



    # only deform object from start time to end time
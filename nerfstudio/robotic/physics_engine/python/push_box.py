# cite from Jatavallabhula and Macklin et al., "gradSim: Differentiable simulation for system identification and visuomotor control", ICLR 2021.

# ode implementation

import numpy as np
import trimesh
import open3d as o3d

from nerfstudio.robotic.physics_engine.python.collision_detection import collision_detection
from nerfstudio.robotic.kinematic.uniform_kinematic import *


from scipy.spatial.transform import Rotation as R

import pypose as pp
from nerfstudio.robotic.export_util.urdf_utils.urdf_helper import find_lower_plane_center

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


    # add support force 



    
    # Calculate the translation vector from fulcrum to the new center of mass
    translation_vector = new_center_of_mass - fulcrum
    
    return new_rotation_matrix, translation_vector

def concat_transformation(rotation_matrix, translation_vector):
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix



def find_object_center(corners):
    """
    Find the center of the object (centroid of the bounding box).
    
    :param corners: List of 8 corners of the bounding box. Each corner is a tuple (x, y, z).
    :return: Tuple representing the center of the object (x, y, z).
    """
    # Calculate the center of the object (mean of all corners)
    object_center = np.mean(corners, axis=0)
    
    return object_center




def simulate_support_force(mass, gravity=9.81, Rot_vec=np.zeros(3)):
    """
    Simulates the support force of the ground on an object.

    Parameters:
    mass (float): Mass of the object in kilograms.
    gravity (float): Acceleration due to gravity in m/s^2. Default is 9.81.
    R (numpy.ndarray): 3x3 rotation matrix. Default is the identity matrix.

    Returns:
    numpy.ndarray: The support force vector.
    """
    # Define the gravity force vector
    F_gravity = np.array([-mass * gravity, 0, -mass * gravity])
    F_gravity_rotated=np.zeros(3)
    # Rotate the gravity force vector using the rotation matrix R
    F_gravity_rotated[2] = F_gravity[2]*np.cos(np.deg2rad(Rot_vec[0]))
    F_gravity_rotated[0] = F_gravity[2]*np.sin(np.deg2rad(Rot_vec[0]))

    

    # The support force is equal in magnitude but opposite in direction to the gravity force and since center of mass and fulcrum are the changed it can be decomposed as the vector.
    F_support = F_gravity_rotated-F_gravity
        
    return F_support

def compute_change_in_position(mass, force, delta_time, initial_velocity=np.zeros(3), initial_position=np.zeros(3)):
    """
    Computes the change in position of an object based on mass, force vector, and change of time.

    Parameters:
    mass (float): Mass of the object in kilograms.
    force (numpy.ndarray): Force vector acting on the object in newtons.
    delta_time (float): Change in time in seconds.
    initial_velocity (numpy.ndarray): Initial velocity vector of the object in m/s. Default is zero.
    initial_position (numpy.ndarray): Initial position vector of the object in meters. Default is zero.

    Returns:
    numpy.ndarray: The new position vector after the given time interval.
    """
    # Compute the acceleration vector
    acceleration = force / mass
    
    # Compute the change in velocity
    delta_velocity = acceleration * delta_time
    
    # Compute the change in position
    delta_position = initial_velocity * delta_time + 0.5 * acceleration * delta_time**2
    
    # Compute the new position

    return delta_position





def interpolate_position_z(angle, mid_position, max_position=-0.01):
    """
    Interpolates a position value based on the input angle, mid_position, and max_position.
    
    Parameters:
    angle (float): The input angle in degrees, ranging from 0 to 90.
    mid_position (float): The position value at 45 degrees.
    max_position (float): The maximum position value at 90 degrees (default is 0.007).
    
    Returns:
    float: The interpolated position value.
    """
    
    if angle <= 45:
        # Normalize the angle to the range [0, 1]
        normalized_angle = angle / 45.0
        # Apply sine-based interpolation from [0, 1] to [0, mid_position]
        interpolated_value = mid_position * np.sin(np.pi / 2 * normalized_angle)
    else:
        # Normalize the angle to the range [0, 1]
        normalized_angle = (angle - 45) / 45.0
        # Apply sine-based interpolation from [0, 1] to [mid_position, max_position]
        interpolated_value = mid_position + (max_position - mid_position) * np.sin(np.pi / 2 * normalized_angle)
    
    return interpolated_value


def interpolate_position_x(angle, mid_position, max_position=0.035):
    """
    Interpolates a position value based on the input angle, mid_position, and max_position.
    
    Parameters:
    angle (float): The input angle in degrees, ranging from 0 to 90.
    mid_position (float): The position value at 45 degrees.
    max_position (float): The maximum position value at 90 degrees (default is 0.007).
    
    Returns:
    float: The interpolated position value.
    """
    
    if angle <= 45:
        # Normalize the angle to the range [0, 1]

        # Apply sine-based interpolation from [0, 1] to [0, mid_position]
        interpolated_value = mid_position + abs(max_position - mid_position) * np.sin(np.pi / 2 * angle)
    else:
        # Normalize the angle to the range [0, 1]
        normalized_angle = (angle - 45) / 45.0
        # Apply sine-based interpolation from [0, 1] to [mid_position, max_position]
    # normalized_angle = angle 
        interpolated_value = mid_position + abs(max_position - mid_position) * np.sin(np.pi / 2 * normalized_angle)
    
    return interpolated_value

def example_push_raw(dt):
       # Box properties
    mass = 10  # mass of the box
    # dimensions = [1.0, 1.0, 1.0]  # dimensions of the box (width, height, depth)
    center_of_mass=np.array([0,0,0])
    # for push case:
    from pathlib import Path
    object_mesh_path=Path('dataset/push_box/part/object/box_convex.obj') # id 8 

    mesh_object = trimesh.load(object_mesh_path)


    bb=mesh_object.bounding_box_oriented

    corners=trimesh.bounds.corners(mesh_object.bounding_box_oriented.bounds)

    recenter_vector=find_lower_plane_center(corners)

    

    # mesh_object.apply_translation(-recenter_vector)


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

    return recenter_vector,new_rotation_matrix, translation_vector,simulation_position

def example_push(dt):
    """
    This is just a simple implementation of simulation, we will add the full implementation later with backward 
    """
       # Box properties
    mass = 10  # mass of the box in grams
    # dimensions = [1.0, 1.0, 1.0]  # dimensions of the box (width, height, depth)
    center_of_mass=np.array([0,0,0])

    from pathlib import Path
    object_mesh_path=Path('dataset/push_box/part/object/box_convex.obj') # id 8 

    mesh_object = trimesh.load(object_mesh_path)


    bb=mesh_object.bounding_box_oriented

    corners=trimesh.bounds.corners(mesh_object.bounding_box_oriented.bounds)


    recenter_vector=find_object_center(corners)

    mesh_object.apply_translation(-recenter_vector)


    dimensions=bb.extents

    bb_recentered=mesh_object.bounding_box_oriented
    center_of_mass=bb_recentered.centroid
    # Compute the moment of inertia
    moment_of_inertia = compute_moment_of_inertia(mass, dimensions)
    force=np.array([0.20,0,0])

    # raw
    # simulation_position=np.array([0.0, 0.0, 0.013])
    simulation_position=np.array([0.0, 0.0, 0.013])*0.4
    # Given torque applied (for example purposes)
    force_position = center_of_mass+ simulation_position # torque vector
    # force_position = center_of_mass # torque vector
    torque=compute_torque(force, force_position)  # torque vector
    # Time step for integration
    dt=dt+0.5

    # Compute angular acceleration
    angular_acceleration = compute_angular_acceleration(torque, moment_of_inertia)

    # Integrate angular acceleration to get angular velocity
    angular_velocity = integrate_angular_acceleration(angular_acceleration, dt)

    fulfrum_position=np.array([0,0,0]) # the position of the fulcrum
    

    # Position vector of the point where you want to compute the linear velocity
    position_vector = force_position- fulfrum_position

    linear_velocity = compute_linear_velocity(angular_velocity, position_vector)







    new_rotation_matrix, translation_vector=leverage_simulation(mesh_object,center_of_mass,fulfrum_position,linear_velocity,angular_velocity,dt)

    # the value need to be optimized by the backward function
    # raw
    # adaptive_vector=np.array([0.0, 0.0, 0.0])
  
    rotation_vector=R.from_matrix(new_rotation_matrix).as_euler('yzx', degrees=True)

    F_supported=simulate_support_force(mass, gravity=9.81, Rot_vec=rotation_vector)


    adaptive_vector=compute_change_in_position(mass, F_supported, dt)/1000 # from kg to g
    adaptive_vector[0]=interpolate_position_x(rotation_vector[0], adaptive_vector[0])
    adaptive_vector[2]=interpolate_position_z(rotation_vector[0], adaptive_vector[2])
    trans=concat_transformation(new_rotation_matrix, translation_vector)




    eye_rotate=np.eye(3)

    back=concat_transformation(eye_rotate, -simulation_position)

    recenter_back_vector=recenter_vector



    return recenter_vector,new_rotation_matrix, translation_vector,simulation_position,adaptive_vector







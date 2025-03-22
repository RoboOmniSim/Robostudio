import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import trimesh


from nerfstudio.robotic.kinematic.uniform_kinematic import *
# this aims to use the physics equation to get the interpolated transformation matrix of each joint and object, then trace it back to the kinematic



def transformation_2_quaternion(transformation):
    """
    Convert transformation matrix to quaternion.
    """
    # Get the rotation matrix
    rotation_matrix = transformation[:3, :3]
    # Convert the rotation matrix to quaternion
    quaternion = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(rotation_matrix)
    return quaternion

def extract_angles_from_transformation(T, a, alpha, d):
    """
    Trace of forward kinematic
    
    """


    theta = np.zeros(6)
    
    # Extract the end-effector position and orientation from T
    R = T[:3, :3]
    P = T[:3, 3]
    
    # Assume T0_6 is the transformation matrix from base to the end effector
    # We'll need to decompose this into the product of individual transformations T1_0, T2_1, ..., T6_5

    # Step 1: Solve for the first joint angle, θ1
    theta[0] = np.arctan2(P[1], P[0])
    
    # Step 2: Use the value of θ1 to compute the position of the second joint and solve for θ2
    # Define the first transformation matrix T1_0 based on θ1
    T1_0 = np.array([
        [np.cos(theta[0]), -np.sin(theta[0]) * np.cos(alpha[0]), np.sin(theta[0]) * np.sin(alpha[0]), a[0] * np.cos(theta[0])],
        [np.sin(theta[0]), np.cos(theta[0]) * np.cos(alpha[0]), -np.cos(theta[0]) * np.sin(alpha[0]), a[0] * np.sin(theta[0])],
        [0, np.sin(alpha[0]), np.cos(alpha[0]), d[0]],
        [0, 0, 0, 1]
    ])
    
    T0_1 = np.linalg.inv(T1_0)
    T2_6 = T0_1 @ T
    
    P2_6 = T2_6[:3, 3]
    P2_6_proj = np.sqrt(P2_6[0]**2 + P2_6[1]**2)
    theta[1] = np.arctan2(P2_6[2] - d[1], P2_6_proj) - np.arctan2(a[2], d[2])
    
    # Step 3: Compute the third joint angle, θ3
    T2_1 = np.array([
        [np.cos(theta[1]), -np.sin(theta[1]) * np.cos(alpha[1]), np.sin(theta[1]) * np.sin(alpha[1]), a[1] * np.cos(theta[1])],
        [np.sin(theta[1]), np.cos(theta[1]) * np.cos(alpha[1]), -np.cos(theta[1]) * np.sin(alpha[1]), a[1] * np.sin(theta[1])],
        [0, np.sin(alpha[1]), np.cos(alpha[1]), d[1]],
        [0, 0, 0, 1]
    ])
    
    T0_2 = T1_0 @ T2_1
    T3_6 = np.linalg.inv(T0_2) @ T
    
    P3_6 = T3_6[:3, 3]
    theta[2] = np.arctan2(P3_6[1], P3_6[0])
    
    # Repeat similar steps for θ4, θ5, and θ6 using the same pattern
    # Calculate the transformation matrices for each joint and solve for the corresponding angles

    # Step 4: Compute the fourth joint angle, θ4
    T3_2 = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]) * np.cos(alpha[2]), np.sin(theta[2]) * np.sin(alpha[2]), a[2] * np.cos(theta[2])],
        [np.sin(theta[2]), np.cos(theta[2]) * np.cos(alpha[2]), -np.cos(theta[2]) * np.sin(alpha[2]), a[2] * np.sin(theta[2])],
        [0, np.sin(alpha[2]), np.cos(alpha[2]), d[2]],
        [0, 0, 0, 1]
    ])
    
    T0_3 = T0_2 @ T3_2
    T4_6 = np.linalg.inv(T0_3) @ T
    
    P4_6 = T4_6[:3, 3]
    theta[3] = np.arctan2(P4_6[2], -P4_6[0])

    # Step 5: Compute the fifth joint angle, θ5
    T4_3 = np.array([
        [np.cos(theta[3]), -np.sin(theta[3]) * np.cos(alpha[3]), np.sin(theta[3]) * np.sin(alpha[3]), a[3] * np.cos(theta[3])],
        [np.sin(theta[3]), np.cos(theta[3]) * np.cos(alpha[3]), -np.cos(theta[3]) * np.sin(alpha[3]), a[3] * np.sin(theta[3])],
        [0, np.sin(alpha[3]), np.cos(alpha[3]), d[3]],
        [0, 0, 0, 1]
    ])
    
    T0_4 = T0_3 @ T4_3
    T5_6 = np.linalg.inv(T0_4) @ T
    
    P5_6 = T5_6[:3, 3]
    theta[4] = np.arctan2(P5_6[1], P5_6[2])
    
    # Step 6: Compute the sixth joint angle, θ6
    T5_4 = np.array([
        [np.cos(theta[4]), -np.sin(theta[4]) * np.cos(alpha[4]), np.sin(theta[4]) * np.sin(alpha[4]), a[4] * np.cos(theta[4])],
        [np.sin(theta[4]), np.cos(theta[4]) * np.cos(alpha[4]), -np.cos(theta[4]) * np.sin(alpha[4]), a[4] * np.sin(theta[4])],
        [0, np.sin(alpha[4]), np.cos(alpha[4]), d[4]],
        [0, 0, 0, 1]
    ])
    
    T0_5 = T0_4 @ T5_4
    T6_6 = np.linalg.inv(T0_5) @ T
    
    P6_6 = T6_6[:3, 3]
    theta[5] = np.arctan2(P6_6[0], P6_6[2])
    
    return theta



def adapt_interpolation_value(transformation,objects, output_file,experiment_type):
    """
    retrive the joint angles from the transformation matrix
    """
    movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file,experiment_type)

    angles = []
    for i in range(6):
        angles.append(extract_angles_from_transformation(transformation[i], a[i], alpha[i], d[i]))


    return angles





# extrapolation of the transformation matrix

def extrapolate(edited_trajectory,fps_rate=1):
    """
    This part aims to extrapolate the edited trajectroy
    
    input: edited_trajectory 

    fps_rate: interpolate frame_rate compare to default fps 10

    
    """
    start_angle=edited_trajectory[0]
    end_angle=edited_trajectory[1]
    interpolated_angles=linear_interpolation_angle(start_angle,end_angle,fps_rate)
    return interpolated_angles


# linear interpolation of the joint angle
def linear_interpolation_angle(start_angles, end_angles, num_points):
    """
    Perform linear interpolation between two sets of joint angles for a 6-DOF robot.

    Args:
    start_angles (np.array): Starting joint angles, a 1x6 numpy array.
    end_angles (np.array): Ending joint angles, a 1x6 numpy array.
    num_points (int): Number of interpolation points.

    Returns:
    np.array: Interpolated joint angles, a num_points x 6 numpy array.
    """
    # Create an array of interpolation factors
    t = np.linspace(0, 1, num_points)
    
    # Interpolate each joint angle
    interpolated_angles = np.outer(t, end_angles - start_angles) + start_angles
    
    return interpolated_angles


from scipy.spatial.transform import Rotation as R

def linear_interpolation_body(x0, x1, t0, t1, t):
    return x0 + (t - t0) / (t1 - t0) * (x1 - x0)

def slerp(q0, q1, t0, t1, t):
    alpha = (t - t0) / (t1 - t0)
    return R.slerp(alpha, R.from_quat([q0, q1]))


def interpolate_body(x0, x1, v0, v1, q0, q1, omega0, omega1, t0, t1, t):
    x_t = linear_interpolation_body(x0, x1, t0, t1, t)
    v_t = linear_interpolation_body(v0, v1, t0, t1, t)
    q_t = slerp(q0, q1, t0, t1, t).as_quat()
    omega_t = linear_interpolation_body(omega0, omega1, t0, t1, t)
    

    # example uses
    # # Given states at t0 and t1
    # t0, t1 = 0, 1
    # x0, x1 = np.array([0, 0, 0]), np.array([1, 1, 1])
    # v0, v1 = np.array([0, 0, 0]), np.array([1, 1, 1])
    # q0, q1 = [1, 0, 0, 0], [0, 1, 0, 0]  # Example quaternions
    # omega0, omega1 = np.array([0, 0, 0]), np.array([1, 1, 1])

    # # Interpolated state at time t
    # t = 0.5
    # x_t = linear_interpolation_body(x0, x1, t0, t1, t)
    # v_t = linear_interpolation_body(v0, v1, t0, t1, t)
    # q_t = slerp(q0, q1, t0, t1, t).as_quat()
    # omega_t = linear_interpolation_body(omega0, omega1, t0, t1, t)
    # def linear_interpolation_rigid_object

    return x_t, v_t, q_t, omega_t



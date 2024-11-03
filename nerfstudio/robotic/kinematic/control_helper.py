

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import trimesh


def create_transformation_matrix_mdh(theta, a, alpha, d):
    """
    Constructs the transformation matrix from the joint parameters using the MDH convention.


    # all radians
    """


    # a is z axis
    # d is x axis
    # theta extra move
    # alpha is the rotation of axis

    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])
    return T

def create_transformation_matrix_dh(theta, d, a, alpha):
    """
    Returns the transformation matrix using DH parameters.
    
    Parameters:
    theta: Joint angle (in radians)
    d: Offset along the previous z-axis
    a: Length of the common normal (link length)
    alpha: Twist angle (in radians)

    Returns:
    A 4x4 numpy array representing the transformation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])




def create_transformation_matrix_mdh_gripper_euler(theta, a, alpha, d):


    """
    Constructs the transformation matrix from the gripper parameters using the MDH convention but under euler angle.

    This is not differentiable but very suit for the gripper control under coordinate transformation

    the original is t^(i-1)_(i) = tans(x,a_(i-1))*rot(x,alpha
    _(i-1))*

    replace to t^(i-1)_(i) = tans(z,a_(i-1))*rot(z,alpha
    _(i-1))*

    # all radians
    """




    T=np.eye(4)



    return T



def create_transformation_matrix_mdh_gripper(theta, a, alpha, d):

    """
    Constructs the transformation matrix from the gripper parameters using the MDH convention.
    the original is t^(i-1)_(i) = tans(x,a_(i-1))*rot(x,alpha_(i-1))*trans(z,d_i)*rot(z,theta_i)

    replace to t^(i-1)_(i) = tans(z,a_(i-1))*rot(z,alpha_(i-1))*trans(y,d_i)*rot(z,theta_i)

    # all radians
    """


    # a is z axis
    #d is x axis
    # theta extra move
    #alpha is the rotation of axis


    Rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

    # Translation along the x-axis by a_n_minus_1
    Trans_x = np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Rotation around the z-axis by theta_n
    Rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Translation along the y-axis by d_n
    Trans_y = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, d],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Compute the combined transformation matrix

    # T = Trans_x @ Rot_x @ Trans_y @ Rot_z
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, -d*np.sin(theta)+a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), d * np.cos(alpha)*np.cos(theta)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.sin(alpha)*np.cos(theta)],
        [0, 0, 0, 1]
    ])
    return T







def create_transformation_matrix_dh_gripper(theta, a, alpha, d):

    """
    Constructs the transformation matrix from the gripper parameters using the MDH convention.
    the original is t^(i-1)_(i) = tans(x,a_(i-1))*rot(x,alpha_(i-1))*trans(z,d_i)*rot(z,theta_i)

    replace to t^(i-1)_(i) = tans(z,a_(i-1))*rot(z,alpha_(i-1))*trans(y,d_i)*rot(z,theta_i)

    # all radians
    """


    # a is z axis
    #d is x axis
    # theta extra move
    #alpha is the rotation of axis


    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])





# Function to create a transformation matrix using DH parameters
def create_transformation_matrix(theta, a, alpha, d):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])




def reflect_x_axis(T):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    R_x = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_reflected = R_x @ T @ R_x
    return T_reflected

def reflect_y_axis(T):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    R_x = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_reflected = R_x @ T @ R_x
    return T_reflected

def reflect_z_axis(T):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    R_x = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    T_reflected = R_x @ T @ R_x
    return T_reflected

def reflect_x_axis_only(T):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    R_x = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])




    T_reflected =  T @ R_x.T
    return T_reflected

def reflect_y_axis_only(T):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    R_x = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])

    R_y = np.array([
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])



    T_reflected = R_y @ T @ R_y
    return T_reflected

def reflect_z_axis_only(T):
    """
    Reflect the transformation matrix along the x-axis.

    Parameters:
    T (numpy.ndarray): The original transformation matrix 4x4

    Returns:
    numpy.ndarray: The reflected transformation matrix 4x4
    """
    R_x = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])

    R_y = np.array([
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])



    T_reflected = R_y @ T @ R_y
    return T_reflected


def create_transformation_matrix_mdh_gripper_reflect_x_coordinate(theta, a, alpha, d):

    """
    Constructs the transformation matrix from the gripper parameters using the MDH convention.
    the original is t^(i-1)_(i) = tans(x,a_(i-1))*rot(x,alpha_(i-1))*trans(z,d_i)*rot(z,theta_i)

    replace to t^(i-1)_(i) = tans(z,a_(i-1))*rot(z,alpha_(i-1))*trans(y,d_i)*rot(z,theta_i)

    # all radians
    """


    # a is z axis
    #d is x axis
    # theta extra move
    #alpha is the rotation of axis
    Rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

    # Translation along the x-axis by a_n_minus_1
    Trans_x = np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Rotation around the z-axis by theta_n
    Rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Translation along the y-axis by d_n
    Trans_y = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, d],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Compute the combined transformation matrix

    T = Trans_x @ Rot_x @ Trans_y @ Rot_z


    # T = np.array([
    #     [np.cos(theta), np.sin(theta), 0, d*np.sin(theta)-a],
    #     [-np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), d * np.cos(alpha)*np.cos(theta)],
    #     [-np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.sin(alpha)*np.cos(theta)],
    #     [0, 0, 0, 1]
    # ])

    return T


def transformation_from_t0_to_t1(T0_i_t0, T0_i_t1):
    # Calculate the inverse of the transformation matrix at time_0
    T0_i_t0_inv = np.linalg.inv(T0_i_t0)
    
    # Calculate the transformation matrix from time_0 to time_1
    Tt0_t1 = np.dot(T0_i_t0_inv, T0_i_t1)
    
    return Tt0_t1



def read_txt_file(filename):

    """
    read file from control output of ROS
    
    """

    data = []
    with open(filename, 'r') as file:
        # Temporary storage for current record
        record = {}
        for line in file:
            line = line.strip()
            if line.startswith('Time:'):
                # Save the previous record if it exists
                if record:
                    data.append(record)
                # Start a new record
                record = {'Time': int(line.split()[1])}
            elif line.startswith('Joint Names:'):
                names = line[len('Joint Names:'):].strip().strip('[]')
                record['Joint Names'] = [name.strip().strip("'") for name in names.split(',')]
            elif line.startswith('Joint Positions:'):
                positions = line[len('Joint Positions:'):].strip().strip('()')
                record['Joint Positions'] = tuple(float(num) for num in positions.split(','))
        # Append the last record if file does not end with a new line
        if record:
            data.append(record)
    return data
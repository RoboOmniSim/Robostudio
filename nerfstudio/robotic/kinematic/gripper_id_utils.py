
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import trimesh



from nerfstudio.robotic.physics_engine.collision_detection import collision_detection
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.kinematic.control_helper import *

def load_gripper_control(output_file,experiment_type,start_time=100,end_time=200):
    """
    Load the gripper control data from the given path.
    """
    if experiment_type=='push_bag':
        # Load the gripper control data for the push box experiment
        movement_angle_state = read_txt_file(output_file) # radians
    elif experiment_type=='novelpose':
        # Load the gripper control data for the pick and place experiment
        movement_angle_state = read_txt_file(output_file) # radians
    elif experiment_type=='grasp':
        # Load the gripper control data for the open and close experiment
        movement_angle_state = read_txt_file(output_file) # radians
    elif experiment_type=='grasp_object':
        # Load the gripper control data for the push box experiment
        movement_angle_state = read_txt_file(output_file) # radians
    else:
        # Load the gripper control data for the default experiment
        movement_angle_state = read_txt_file(output_file) # radians

    joint_angles_degrees_gripper = np.zeros(5)
    a_gripper = np.zeros(5)
    alpha_gripper = np.zeros(5)
    d_gripper = np.zeros(5)


    joint_angles_degrees_gripper[1]=1.83759# for left down
    joint_angles_degrees_gripper[2]=2.8658# for left up
    joint_angles_degrees_gripper[3]=1.83759# for right down
    joint_angles_degrees_gripper[4]=2.8658# for right up

    a_gripper[0]=0
    a_gripper[1]=-0.03853 
    a_gripper[2]=0.041998
    a_gripper[3]=-0.03853
    a_gripper[4]=0.041998

    alpha_gripper[1]=np.pi/2
    alpha_gripper[2]=0
    alpha_gripper[3]=np.pi/2
    alpha_gripper[4]=0

    d_gripper[0]=0.11
    d_gripper[1]=0
    d_gripper[2]=0
    d_gripper[3]=0
    d_gripper[4]=0

    gripper_control_mode = np.zeros(len(movement_angle_state))


    dof_limit=20
    degrees = np.linspace(0, dof_limit, num=int((end_time-start_time)*10))  # 10 fps

    # Convert degrees to radians
    radians = np.deg2rad(degrees)
    # write a linear interpolation from start angle to end angle
    for i in range(start_time,end_time):
        gripper_control_mode[i]=radians[i-start_time]
    # gripper_control_mode[100]=1 # the start time of gripper



    return gripper_control_mode,joint_angles_degrees_gripper, a_gripper, alpha_gripper, d_gripper





def gripper_divide(gripper_mesh,scale):
    """
    Divide the gripper mesh into 5 parts: left_down,left_up, and right_down,right_up,center.
    """
    # Get the bounding box of the gripper
    bbox = gripper_mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    center = (min_bound + max_bound) / 2

    # Get the vertices of the gripper
    vertices = np.asarray(gripper_mesh.vertices)
    left_vertices = vertices[vertices[:, 0] < center[0]]
    right_vertices = vertices[vertices[:, 0] >= center[0]]

    # Get the faces of the gripper
    faces = np.asarray(gripper_mesh.triangles)
    left_faces = []
    right_faces = []
    for i, face in enumerate(faces):
        if np.all(vertices[face][:, 0] < center[0]):
            left_faces.append(face)
        elif np.all(vertices[face][:, 0] >= center[0]):
            right_faces.append(face)

    # Create the left and right gripper meshes
    left_gripper_mesh = o3d.geometry.TriangleMesh()
    left_gripper_mesh.vertices = o3d.utility.Vector3dVector(left_vertices)
    left_gripper_mesh.triangles = o3d.utility.Vector3iVector(left_faces)
    right_gripper_mesh = o3d.geometry.TriangleMesh()
    right_gripper_mesh.vertices = o3d.utility.Vector3dVector(right_vertices)
    right_gripper_mesh.triangles = o3d.utility.Vector3iVector(right_faces)

    return left_gripper_mesh_0,left_gripper_mesh_1, right_gripper_mesh_0,right_gripper_mesh_1,center_mesh



def reassign_id(mesh):
    """
    Reassign the semantic IDs of the gripper. The IDs are assigned as follows:
    if no objects
    - center: 7
    - object: 8
    - left_down: 9
    - left_up: 10
    - right_down: 11
    - right_up: 12
    - base: 13

    if objects
    - center: 8+num_objects
    - left_down: 9+num_objects
    - left_up: 10+num_objects
    - right_down: 11+num_objects
    - right_up: 12+num_objects
    # object insert between 7 and 8, ie the gripper center and base


    

    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    num_vertices = len(vertices)
    num_faces = len(faces)

    # Create
    new_vertices = np.arange(num_vertices).reshape(-1, 1)
    id=0

    return new_vertices, new_faces


def gripper_control(control_mode,gripper_mesh,object_mesh,timestamp):
    """
    
    control_mode are the open state of the gripper and the close state of the gripper, it should be a time array that has same index with timestamp
    if the gripper is change from open to close, then the gripper is move from open angle to close angle.
    if there is collision detected, then we need to stop the change of joint angle in this moment


    0 is close
    1 is open



    gripper_mesh is the mesh of the gripper

    object_mesh is the mesh of the object

    timestamp is the time array that has the same length with control_mode
    



    """

    joint_angle_list=[[]]*len(timestamp)
    initial_angle=0 # the initial angle of the gripper when open
    for time in timestamp:
        joint_angle_list[time]=initial_angle
        if control_mode[time]==0:
            # Close the gripper
            if control_mode[time-1]==1:
                # Move the gripper from open to close
                joint_angle_compute, T2_0=forward_gripper_mdh(gripper_mesh)
                R2_0=T2_0[:3,:3]
                p2_0=T2_0[:3,3]
                output_mesh = gripper_mesh@ R2_0.T + p2_0
                if collision_detection(output_mesh,object_mesh):
                    # Stop the movement

                    joint_angle_list[time]=joint_angle_compute
                else:
                    # keep the gripper close
                    joint_angle_list[time]=joint_angle_list[time-1]
                

            else:
                # keep the gripper open
                joint_angle_list[time]=initial_angle

    return joint_angle_list
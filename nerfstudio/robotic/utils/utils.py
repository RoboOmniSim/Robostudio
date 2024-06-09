import os
import numpy as np
import open3d as o3d

import scipy

import matplotlib.pyplot as plt
import trimesh



def load_transformation_package(path):
    """
    Load transformation package from a folder.
    """
    # Load transformation matrices




    # save different link transformation to different index
    tf_matrices = dict()
    for file in os.listdir(path):
        if file.endswith("_tf.txt"):
            link = file.split("_")[0]
            tf_matrices[link] = np.loadtxt(os.path.join(path, file))



    return tf_matrices


def load_txt_bbox(file_path):
    """
    Load bbox package from a txt.
    """

    with open(file_path, 'r') as file:
        data = file.read()

    # Parse the data into a list of numpy arrays
    data = data.strip().split('\n')
    arrays = [np.fromstring(item.strip('[]'), sep=' ') for item in data]
    return arrays


def load_txt(path):
    """
    Load transformation package from a txt.
    """
    # Load transformation matrices

    
    with open(path, 'r') as file:
        lines = file.readlines()

    info = {}
    current_key = None
    current_value = []

    for line in lines:
        line = line.strip()
        if line.endswith('.ply'):
            current_key = line
            current_value = []
            if current_key is not None:
                info[current_key]=current_value
            
        elif line.startswith('['):
            current_value.append(np.fromstring(line.strip('[]'), sep=' '))
        elif line:
            current_value.append(np.array(line.split()))

    return info



def convert_obj_to_ply_scene(path):
    scene_list=[]
    # faces_save_list=[]
     # vertices, faces
    file_name_list_scene=[]
    path_list=os.listdir(path)
    path_list.sort()
    for file in path_list:
        if file.endswith('.obj'):
            obj_path = os.path.join(path, file)
            ply_path = os.path.join(path, file.replace('.obj', '.ply'))
            

            # Load the OBJ file
            mesh = trimesh.load(obj_path)
            # mesh.apply_obb()
            # vertices = mesh.vertices
            # faces = mesh.faces
            # # Export the mesh as a PLY file
            # # mesh.export(ply_path, file_type='ply')
            # vertices_save_list.append(vertices)
            # faces_save_list.append(faces)
            scene_list.append(mesh)
        file_name_list_scene.append(file)
    return scene_list,file_name_list_scene








def relative_tf_to_global(tf_matrices):
    """
    Convert relative transformation matrices to global transformation matrices.


    # only work for 6 links model, need to change for different kinematics
    """
    global_tf_matrices = np.zeros((len(tf_matrices),tf_matrices['Link1'].shape[0], 4))

    # edit this for different kinematics
    for i in range(len(tf_matrices)):
        if i == 0:
            global_tf_matrices[0,:]=tf_matrices['Link1']
        elif i == 1:
            global_tf_matrices[1,:]=tf_matrices['Link2']
        elif i == 2:
            global_tf_matrices[2,:]=tf_matrices['Link3']
        elif i == 3:
            global_tf_matrices[3,:]=tf_matrices['Link4']
        elif i == 4:
            global_tf_matrices[4,:]=tf_matrices['Link5']
        elif i == 5:
            global_tf_matrices[5,:]=tf_matrices['Link6']
    

    
    global_tf_matrices = global_tf_matrices.reshape((len(tf_matrices), int(int(tf_matrices['Link1'].shape[0])/4), 4, 4))

    rotation_original = global_tf_matrices[:,0,:3, :3]
    translation_original = global_tf_matrices[:,0,:3, 3]

    rotation_deform = global_tf_matrices[:,1:,:3, :3]
    translation_final = global_tf_matrices[:,1:,:3, 3]


    global_original = global_tf_matrices[:,0,:, :]


    global_rela= np.zeros((len(tf_matrices),int(int(tf_matrices['Link1'].shape[0])/4)-1,4,4))


    global_deform = global_tf_matrices[:,1:,:,:]
    for i in range(len(tf_matrices)):
        inv_K=np.linalg.inv(global_original[i,:,:])
        global_rela[i,:,:,:] = global_deform[i,:,:,:] @ inv_K.reshape(1,1,4,4)


    # rotation_rela = global_rela[:,:,:3, :3]
    # translation_rela = global_rela[:,:,:3, 3]

    # process the matrix and save it for each timestamp




    rotation_rela= np.zeros((len(tf_matrices),int(int(tf_matrices['Link1'].shape[0])/4)-1,3,3))


    
    for i in range(len(tf_matrices)):
        inv_R=np.linalg.inv(rotation_original[i,:,:])
        rotation_rela[i,:,:,:] = rotation_deform[i,:,:,:] @ inv_R.reshape(1,1,3,3)

    translation_final = global_tf_matrices[:,1:,:3, 3]

    translation_rela = np.zeros((len(tf_matrices),int(int(tf_matrices['Link1'].shape[0])/4)-1,3))
    for i in range(len(tf_matrices)):
        translation_original_i=translation_original[i]
        # translation_rela[i,:,:] = translation_final[i,:,:] - rotation_rela[i,:,:,:] @translation_original_i
        translation_rela[i,:,:] = translation_final[i,:,:] - translation_original_i

    translation_rela[np.abs(translation_rela) <= 1e-3] = 0
    rotation_rela[np.abs(rotation_rela) <= 1e-5] = 0
    
    return rotation_rela, translation_rela,rotation_original,translation_original,rotation_deform,translation_final















def update_global_transformations(K_rela, delta_R, delta_T):
    """
    Calculate global transformation matrices for each link.

    Parameters:
    K_rela (list of numpy.ndarray): Initial relative transformation matrices (4x4) for each link.
    delta_R (list of numpy.ndarray): Rotation matrices (3x3) representing changes for each link.
    delta_T (list of numpy.ndarray): Translation vectors (3x1) representing changes for each link.

    Returns:
    list of numpy.ndarray: Updated global transformation matrices for each link.
    """
    # Number of links
    n_links = len(K_rela)

    # Initialize global transformations with the relative transformations
    K_global = K_rela.copy()

    # Update each link's transformation matrix
    for i in range(n_links):
        # Create the transformation matrix for the change
        delta_K = np.eye(4)
        delta_K[:3, :3] = delta_R[i]
        delta_K[:3, 3] = delta_T[i]

        # Update the global transformation of the current link
        if i == 0:
            K_global[i] = np.dot(K_rela[i], delta_K)
        else:
            K_global[i] = np.dot(K_global[i-1], delta_K)

    return K_global
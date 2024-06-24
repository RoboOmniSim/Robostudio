# get semantic labeling by reprojection from 2d sam mask, pre-trained model or manual bbox labeling


from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
import trimesh
import os



def semantic_labeling(mesh,semantic_labeling_path):
    """
    Semantic labeling for the mesh based on the reprojection of 2d mask from the camera to the gaussian and the gaussian-mesh binding
    """
    # Load the semantic labeling
    semantic_labeling = load_txt(semantic_labeling_path)

    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)

    # Get the faces of the mesh
    faces = np.asarray(mesh.triangles)

    # Get the semantic labeling of the vertices
    semantic_labeling_vertices = []
    for vertex in vertices:
        semantic_labeling_vertices.append(semantic_labeling[vertex])

    # Get the semantic labeling of the faces
    semantic_labeling_faces = []
    for face in faces:
        semantic_labeling_faces.append(semantic_labeling[face])

    return semantic_labeling_vertices, semantic_labeling_faces
    
import pypose as pp
def interp_sampling(mask, coord):
    sampled_fine = torch.nn.functional.grid_sample(mask[None], 
                                                   coord[None][None], align_corners=False,
                                                   mode='nearest')
    sampled_fine = sampled_fine[0, :, 0, :].mT
    return sampled_fine

def project_sam_mask(xyz,projection_matrix,view_mat,mask):
    
    # project 2d uv mask to 3d points
    pts_c = pp.mat2SE3(view_mat).Act(xyz)
    homo = projection_matrix @ pts_c # n*3 output
    points = pp.homo2cart(homo) # n*2, range: (-1, 1) for both x and y

    labels = interp_sampling(mask, points)

    # if points is in edge, set it to zero

    # return the semantic mask of each 3d points in world coordinate and same ordering

    return labels   



# def increment_labels(cameras, labels,xyz):

#     # camers 100

#     # xyz 10000*3

#     label_list=[[xyz]]*len(camears)
#     for camera in cameras:
#         mask=load_mask

#         xyz=load_ply
#         projection_matrix,view_mat= camera.extrinsic, intrinsci



#         labels=project_sam_mask(xyz,projection_matrix,view_mat,mask) # share same dimension with xyz

#         # move labels to cpu
#         label_list[i]=labels



#     label_list # 100*10000*1

#     for i in 10000:
#         100 labels, 

#         one hot, or any distribution 
#         set this largest value to the semantic mask of this point.

#     return semantic_mask


# def pixel_trajectory_gs(gs,pixel,traj):
#     traj # pixel*timestamp , optical flow


#     traj # current index, t1 saved value, t0


#     t0_pixel_project_gs map to t1_pixel  # supervision t_0 static projection + depth changes

#     # output new_porjection matrix. static projection matrix + depth changes+global transformation

#     return new_projection_matrix





def load_txt(path):
    pass




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


def pretrained_model_semantic_labeling(mesh,pretrained_model_path):
    """
    Semantic labeling for the mesh based on the pre-trained model.
    """
    # Load the pre-trained model
    pretrained_model = load_txt(pretrained_model_path)

    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)

    # Get the faces of the mesh
    faces = np.asarray(mesh.triangles)

    # Get the semantic labeling of the vertices
    semantic_labeling_vertices = []
    for vertex in vertices:
        semantic_labeling_vertices.append(pretrained_model[vertex])

    # Get the semantic labeling of the faces
    semantic_labeling_faces = []
    for face in faces:
        semantic_labeling_faces.append(pretrained_model[face])

    return semantic_labeling_vertices, semantic_labeling_faces





def manual_bbox_labeling(mesh,bbox_labeling_path):
    """
    Semantic labeling for the mesh based on the manual bounding box labeling.
    """
    # Load the bounding box labeling
    bbox_labeling = load_txt(bbox_labeling_path)

    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)

    # Get the faces of the mesh
    faces = np.asarray(mesh.triangles)

    # Get the semantic labeling of the vertices
    semantic_labeling_vertices = []
    for vertex in vertices:
        semantic_labeling_vertices.append(bbox_labeling[vertex])

    # Get the semantic labeling of the faces
    semantic_labeling_faces = []
    for face in faces:
        semantic_labeling_faces.append(bbox_labeling[face])

    return semantic_labeling_vertices, semantic_labeling_faces
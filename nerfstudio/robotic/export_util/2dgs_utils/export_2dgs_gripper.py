from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
import trimesh
import os



def convert_obj_to_ply(path,num_linkes):

    points_list=[[]]*num_linkes
    file_name_list=[[]]*num_linkes
    face_list=[[]]*num_linkes
    color_list=[[]]*num_linkes
    normals_list=[[]]*num_linkes

     # vertices, faces
    path_list=os.listdir(path)
    path_list.sort()
    i=0
    for file in path_list:
        if file.endswith('.obj'):
            obj_path = os.path.join(path, file)

            obj_data = o3d.io.read_triangle_mesh(obj_path)
            # points=np.asarray(ply_data.vertices)
            # ply_data = o3d.io.read_point_cloud(ply_path)
            vertices=np.asarray(obj_data.vertices)
            face=np.asarray(obj_data.triangles)
            color=np.asarray(obj_data.vertex_colors)
            normals=np.asarray(obj_data.vertex_normals)
            # Load the OBJ file

            points_list[i]=vertices
            face_list[i]=face
            color_list[i]=color
            normals_list[i]=normals

            
            file_name_list[i]=file
            i+=1
    return points_list,face_list,color_list,normals_list,file_name_list


def apply_transformation(gs_mesh_list_vertices,gs_mesh_list_faces,num_linkes):
    urdf_mesh_vertices_list=[[]]*num_linkes
    urdf_mesh_faces_list=[[]]*num_linkes

    raw_mesh_vertices_list=[[]]*num_linkes
    raw_mesh_faces_list=[[]]*num_linkes
    R_raw2ori_list=[[]]*num_linkes   
    T_raw2ori_list=[[]]*num_linkes
    R_ori2raw_list=[[]]*num_linkes
    T_ori2raw_list=[[]]*num_linkes
    Matrix_reorientated_list=[[]]*num_linkes
    for i in range(num_linkes):
        mesh=trimesh.Trimesh(vertices=gs_mesh_list_vertices[i],faces=gs_mesh_list_faces[i])
        raw_mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces)


        raw_mesh_vertices_list[i]=raw_mesh.vertices
        raw_mesh_faces_list[i]=raw_mesh.faces
         
        if i==7:
            matrix=mesh.apply_obb()  # recenter
        else:
            matrix=mesh.apply_obb()  # recenter 
        # mesh.export(f'test_{i}.ply', file_type='ply')  #
        # if usepointcloud:
        #     reshaped_vertices=mesh.vertices.reshape(3,-1)
        #     reshaped_raw_vertices=raw_mesh.vertices.reshape(3,-1)

        #     R_raw2ori,T_raw2ori=rigid_transform_3D(reshaped_raw_vertices, reshaped_vertices)

        #     R_ori2raw,T_ori2raw=rigid_transform_3D(reshaped_vertices, reshaped_raw_vertices)
        # else:
        R_raw2ori=0
        T_raw2ori=0
        R_ori2raw=0
        T_ori2raw=0
        R_raw2ori_list[i]=R_raw2ori
        T_raw2ori_list[i]=T_raw2ori
        R_ori2raw_list[i]=R_ori2raw
        T_ori2raw_list[i]=T_ori2raw
        Matrix_reorientated_list[i]=matrix
        # reset coordinate
        # mesh.apply_transform(coordinate_info[i])

        # reset center by moving the center to the motor engine location 
        # mesh.apply_transform(new_center_list[i])
        urdf_mesh_vertices_list[i]=mesh.vertices
        urdf_mesh_faces_list[i]=mesh.faces
    return urdf_mesh_vertices_list,urdf_mesh_faces_list,raw_mesh_vertices_list,raw_mesh_faces_list,R_raw2ori_list,T_raw2ori_list,R_ori2raw_list,T_ori2raw_list,Matrix_reorientated_list


# gripper recenter 
def save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,color_list,normals_list,save_path,num_linkes,filename_list_used):
    for i in range(num_linkes):
        deform_point=urdf_mesh_vertices_list[i]
        select_faces=urdf_mesh_faces_list[i]
        select_color=color_list[i]
        select_normals=normals_list[i]
        file_name=filename_list_used[i]
          
        output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
        output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
        output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
        # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
        o3d.io.write_triangle_mesh(os.path.join(save_path,file_name),output_mesh)


      
        # mesh.export(os.path.join(save_path,file_name), file_type='obj')
           
if __name__ == '__main__':
    num_linkes=5
    # save_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/recenter_mesh'
    # gs_load_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/gs_mesh/linkage'
    save_path = '/home/lou/Downloads/urdf_2dgs/2dgsmesh/recenter_2dgs'
    output_dir = './cleaned_meshes'

    # sugar nesg
    # gs_load_path='./plyresult'

    gs_load_path = './2dgsmesh/body'  # 2dgs mesh

    gripper_load_path = '/home/lou/Downloads/urdf_2dgs/2dgsmesh/gripper'  # 2dgs mesh

    # original_load_path='./workspace/original_link' # sugar

    original_load_path = './original_link_nogripper'  # nerfstudio

    point_list, face_list, color_list, normals_list, file_name_list = convert_obj_to_ply(gripper_load_path, num_linkes)

    # get_minimal_oriented_bounding_box

    # o3d.geometry.OrientedBoundingBox.create_from_points  

    urdf_mesh_vertices_list, urdf_mesh_faces_list, raw_mesh_vertices_list, raw_mesh_faces_list, R_raw2ori_list, T_raw2ori_list, R_ori2raw_list, T_ori2raw_list, Matrix_reorientated_list = apply_transformation(point_list, face_list, num_linkes)




    for i in range(len(urdf_mesh_vertices_list)):
        if i == 0:
            vector_0=np.array([0,0,0])
            vertices=urdf_mesh_vertices_list[i]+vector_0
            urdf_mesh_vertices_list[i]=vertices
        elif i ==1:
            vector_1=np.array([0,0,0])
            vertices=urdf_mesh_vertices_list[i]+vector_1
            urdf_mesh_vertices_list[i]=vertices
        elif i ==2:
            vector_2=np.array([0,0,0])
            vertices=urdf_mesh_vertices_list[i]+vector_2
            urdf_mesh_vertices_list[i]=vertices
        elif i ==3:
            vector_3=np.array([0,0,0])
            rotation_fix=np.array([[1,0,0],[0,1,0],[0,0,1]])
            vertices=urdf_mesh_vertices_list[i]+vector_3
            vertices=vertices@rotation_fix
            urdf_mesh_vertices_list[i]=vertices
        elif i ==4:
            vector_4=np.array([0,0,0])
            vertices=urdf_mesh_vertices_list[i]+vector_4
            urdf_mesh_vertices_list[i]=vertices   
        else:
            vertices=urdf_mesh_vertices_list[i]
            urdf_mesh_vertices_list[i]=vertices



    save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,color_list,normals_list,save_path,num_linkes,file_name_list)

    

import numpy as np
import pypose as pp
import os
import open3d as o3d
import trimesh



import argparse
from nerfstudio.robotic.config.raw_config import export_urdf_to_omnisim_config

# run command: python nerfstudio/robotic/export_util/urdf_utils/urdf.py --part_path ./dataset/roboarm2/urdf/2dgs/arm --save_path ./dataset/roboarm2/urdf/2dgs/recenter_mesh --kinematic_info_path ./config_info/kinematic_info.yaml --experiment_type cr3 --scale_factor_gt 1.0 --num_links 8 --original_path dataset/roboarm2/roboarm2/urdf/2dgs/original_link




def apply_cordinate_shift_gt(urdf_mesh_vertices_list,coordinate_info_R,num_linkes):
    urdf_mesh_vertices_list_reori=[[]]*num_linkes
    for i in range(num_linkes):
        re=np.dot(urdf_mesh_vertices_list[i],coordinate_info_R[i])
        
        urdf_mesh_vertices_list_reori[i]=re
    return urdf_mesh_vertices_list_reori



def save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,color_list,normals_list,save_path,num_linkes,filename_list_used):
    for i in range(num_linkes):
        deform_point=urdf_mesh_vertices_list[i]
        select_faces=urdf_mesh_faces_list[i]
        select_color=color_list[i]
        select_normals=normals_list[i]
        file_name=filename_list_used[i]
          
        output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
        output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
        from copy import deepcopy
        select_normals_copy=deepcopy(select_normals)
        output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals_copy)

                
        # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
        o3d.io.write_triangle_mesh(os.path.join(save_path,file_name),output_mesh)



def convert_obj_to_ply_scene(path):
    scene_list=[]
    file_name_list_scene=[]
    path_list=os.listdir(path)
    path_list.sort()
    for file in path_list:
        if file.endswith('.obj'):
            obj_path = os.path.join(path, file)

            

            # Load the OBJ file
            mesh = trimesh.load(obj_path)

            scene_list.append(mesh)
        file_name_list_scene.append(file)
    return scene_list,file_name_list_scene




def compute_bbox_gs(vertices):
     
    bbox = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    return bbox


def compute_center(bbox,num_linkes):
    center_list=[[]]*num_linkes # 7 is the number of linkes

    for i in range(num_linkes):
        center_list[i]=np.array([0,0,0])

    
    for i in range(num_linkes):
        upper_box = bbox[i][1] 
        lower_box = bbox[i][0]
        proportion=upper_box/(np.abs(upper_box)+np.abs(lower_box))
        proportion=np.maximum(proportion,0.005)
        center_list[i]=proportion

    return center_list

def compute_center_move(bbox,num_linkes):
    center_move_list=[[]]*num_linkes

    for i in range(num_linkes):
        center_move_list[i]=np.array([0,0,0])
    
    for i in range(num_linkes):
        # # manual for non-centric motor engine ,sugar case
        if bbox[i][0][2]>0:
            move_rate_x=(abs(bbox[i][0][0])+abs(bbox[i][1][1]))/2
            move_rate_y=(abs(bbox[i][0][1])+abs(bbox[i][1][1]))/2
            move_rate_z=(abs(bbox[i][0][2])+abs(bbox[i][1][2]))/2
            center_move_list[i]=np.array([move_rate_x,move_rate_y,move_rate_z])

        half_center=(abs(bbox[i][0][2])+abs(bbox[i][1][2]))/2

        
        if bbox[i][0][2]<0 :
            if (half_center*0.8)>abs(bbox[i][0][2]) :
                move_rate=half_center-abs(bbox[i][0][2]) 
                center_move_list[i]=np.array([0,0,-move_rate])



        
    return center_move_list



def convert_obj_to_ply(path,num_linkes,Urdfinfo):

    points_list=[[]]*num_linkes
    file_name_list=[[]]*num_linkes
    face_list=[[]]*num_linkes
    color_list=[[]]*num_linkes
    normals_list=[[]]*num_linkes
    move_to_center_matrix=[[]]*num_linkes
    bbox_save_list=[]
     # vertices, faces
    path_list=os.listdir(path)
    path_list.sort()
    index=0
    for file in path_list:
        if file.endswith('.obj'):
            obj_path = os.path.join(path, file)

            obj_data = o3d.io.read_triangle_mesh(obj_path)


            triangle_clusters, cluster_n_triangles, cluster_area = obj_data.cluster_connected_triangles()
            

            largest_cluster_idx = np.argmax(cluster_area)
            triangles_to_remove = [i for i, c in enumerate(triangle_clusters) if c != largest_cluster_idx]
            

            obj_data.remove_triangles_by_index(triangles_to_remove)

            obj_data.remove_unreferenced_vertices()

            vertices=np.asarray(obj_data.vertices)
            face=np.asarray(obj_data.triangles)
            color=np.asarray(obj_data.vertex_colors)
            normals=np.asarray(obj_data.vertex_normals)
            # Load the OBJ file

            tri_mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(face),
                                    vertex_normals=np.asarray(normals))
            
            matrix=tri_mesh.apply_obb() 
            
            bbox = tri_mesh.bounding_box.bounds
            bbox_save_list.append(bbox)
            
            if Urdfinfo.use_recenter:
                points_list[index]=tri_mesh.vertices
            elif Urdfinfo.use_recenter == False and Urdfinfo.use_kinematic==False and Urdfinfo.use_backward== False:
                points_list[index]=tri_mesh.vertices
            else:
                points_list[index]=vertices
            
            face_list[index]=face
            color_list[index]=color
            normals_list[index]=normals

            move_to_center_matrix[index]=matrix
            file_name_list[index]=file
            index+=1
    return points_list,face_list,color_list,normals_list,file_name_list,bbox_save_list,move_to_center_matrix



def edit_with_transformation(urdf_mesh_vertices_list,new_center_list):
    for i in range(len(urdf_mesh_vertices_list)):
        vertices=urdf_mesh_vertices_list[i]+new_center_list[i]
        
        urdf_mesh_vertices_list[i]=vertices
    return urdf_mesh_vertices_list

def main():
        parser = argparse.ArgumentParser(description="export urdf recenter part")
        
        # Add arguments
        parser.add_argument('--part_path', type=str, help='path to the full part mesh file')
        parser.add_argument('--save_path', type=str, help='path to save bbox list information and recentered urdf. varies based on experiment type')
        parser.add_argument('--kinematic_info_path', type=str, help='path to load scale and kinematic information. varies based on robotic arm')
        parser.add_argument('--experiment_type', type=str, help='type of the experiment')
        parser.add_argument('--scale_factor_gt', type=float, help='scale factor of the part')
        parser.add_argument('--num_links', type=int, help='number of links in the robotic arm')
        parser.add_argument('--original_path', type=str, help='path to the original part mesh file')
        # scale_factor
        args = parser.parse_args()

        if os.path.exists(args.save_path)==False:
            os.makedirs(args.save_path)




        # load gt urdf
        original_scene_list,original_file_name_list=convert_obj_to_ply_scene(args.original_path)

        num_linkes=args.num_links

        gs_bbox_list=[]
        original_bbox_list=[]
        original_bbox_nonori_list=[]
        # scale diff between the gs_mesh and the original mesh due to the gs training intrinsic variance 
        scale=[[]]*num_linkes



        for original_bbox_asset in original_scene_list:
            
            original_bbox_ori = original_bbox_asset.bounding_box_oriented.bounds
            original_bbox=original_bbox_asset.bounding_box.bounds
            original_bbox_list.append(original_bbox_ori)
            original_bbox_nonori_list.append(original_bbox)
        center_list=compute_center(original_bbox_list,num_linkes)

        

        # after have the new_center_list, we can use it to generate the new mesh
        
        center_move_list=compute_center_move(original_bbox_nonori_list,num_linkes)
        

        Urdfinfo=export_urdf_to_omnisim_config()
        Urdfinfo.setup_params(args.kinematic_info_path)

        use_kinematic=Urdfinfo.use_kinematic
        use_recenter=Urdfinfo.use_recenter
        use_backward=Urdfinfo.use_backward



        points_list,face_list,color_list,normals_list,file_name_list,bbox_save_list,move_to_center_matrix=convert_obj_to_ply(args.part_path,args.num_links,Urdfinfo)
        # Load the part mesh
        bounding_box_corners = bbox_save_list[0]
        
        # this method works for  - + - coordinate system for recenter_vector
        extent_base=bounding_box_corners[1]-bounding_box_corners[0]

        # compare with the base_gt and bounding boxe base

        # test this scale factor 
    
        # Get the coordinates of the lower plane center
        center_vector_compute = bounding_box_corners.mean(axis=0)
        center_vector_compute[2] = bounding_box_corners[0][2]
        center_matrix=np.eye(4)
        center_matrix[:3,3]=center_vector_compute*-1
    
        # adapt 2dgs coordinate to issac
        I=np.identity(3)

        gl23d=np.array([[0, 1, 0],
                        [ 0, 0, 1],
                        [1, 0,  0]])
        coordinate_info=np.concatenate([[gl23d]] * num_linkes, axis=0)
        




        coordinate_info[0,:,:]=np.array([ [0,1,0],
                                        [ 0, 0, -1],
                                        [1,0,0]])@coordinate_info[0,:,:]#0

        coordinate_info[1,:,:]=np.array([ [0, 0, -1],
            [-1,0,0],
            [0,-1,0]
            ]
            )@coordinate_info[1,:,:]#1


        coordinate_info[2,:,:]=coordinate_info[2,:,:]@np.array([ [1,0, 0],    
                                        [  0,-1, 0],
                                        [0, 0,  -1]])#2
        

        coordinate_info[3,:,:]=np.array([[0, 0, 1],   
                                        [ 0, -1, 0],
                                        [1, 0, 0]])@coordinate_info[3,:,:]#3

        coordinate_info[4,:,:]=np.array([[1,0,0],
                                        [0,0,1],
                                        [0,1,0]])@coordinate_info[4,:,:]#4
        
        coordinate_info[5,:,:]=np.array([[0,0,-1],
                                        [1,0,0],
                                        [0,1,0]])@coordinate_info[5,:,:] #5
        

        coordinate_info[6,:,:]=np.array([[0, -1, 0], 
                                        [ 0, 0, -1],
                                        [-1, 0,  0]])@coordinate_info[6,:,:] #
        



        # this is for pytorch3d coordinate
        # coordinate_info[5,:,:]=np.array([[1, 0, 0],
        #                                 [ 0, 0, 1],
        #                                 [0, -1,  0]]) #5
        

        # coordinate_info[6,:,:]=np.array([   [0, 0, 1],
        #                                 [ 0, 1, 0],
        #                                 [-1, 0,  0]])#6
        
        # coordinate_info[0,:,:]=np.array([   [0, 0, 1],
        #                                 [ 0, 1, 0],
        #                                 [-1, 0,  0]])#0
        

        # coordinate_info[3,:,:]=np.array([[0, -1, 0],
        #                                 [ 0, 0, 1],
        #                                 [-1, 0, 0]])#3
        

        # coordinate_info[2,:,:]=np.array([ [0, 1, 0],
        #                                 [ 0, 0, -1],
        #                                 [-1, 0,  0]])#2
        
        
        # coordinate_info[4,:,:]=np.array([  [0, 0, 1],
        #                                 [ 1, 0, 0],
        #                                 [0, 1,  0]])#4
        
        # coordinate_info[1,:,:]=np.array([   [-1, 0, 0],
        #                                 [ 0, -1, 0],
        #                                 [0, 0,  1]])#1
        
        # coordinate_info[7,:,:]=np.array([   [1, 0, 0],
        #                                 [ 0, 0, 1],
        #                                 [0, -1,  0]])#1

        urdf_mesh_vertices_list_reorigt=apply_cordinate_shift_gt(points_list,coordinate_info,num_linkes)

        urdf_bbox_reori_list=[]
        for mesh_vertices in urdf_mesh_vertices_list_reorigt:
            
            urdf_bbox_reori = compute_bbox_gs(mesh_vertices)
            urdf_bbox_reori_list.append(urdf_bbox_reori)
        
        new_center_list=[[]]*num_linkes

        # this is various based on different kinematic
        for i in range(len(center_list)):
            raw=center_list[i]*((abs(urdf_bbox_reori_list[i][0])+abs(urdf_bbox_reori_list[i][1])))-abs(urdf_bbox_reori_list[i][1])
            if center_move_list[i][2]==0:
                # link 2 has outside motor engine
                new_center_list[i]=raw
            else:
                if i==2:
                    new_center_list[i]=np.zeros(3)
                    new_center_list[i][0]=0-0.6395
                    new_center_list[i][1]=0
                    new_center_list[i][2]=0+0.25
                if i==3:
                    new_center_list[i]=np.zeros(3)
                    new_center_list[i][0]=0
                    new_center_list[i][1]=0 # by scale computation, write automatic in future 
                    new_center_list[i][2]=0-0.495
                else:
                    new_center_list[i]=raw





        
        moved_center_vertices_list=edit_with_transformation(urdf_mesh_vertices_list_reorigt,new_center_list)




        # 2dgs coordinate extra fix




        tdgs_coordinate_fix=np.array([[1, 0, 0],
                                    [ 0, 1, 0],
                                    [0, 0,  1]])
        
        tdgs_coordinate_info=np.concatenate([[tdgs_coordinate_fix]] * num_linkes, axis=0)


        tdgs_coordinate_info[0,:,:]=np.array([ [1, 0, 0],
                                        [ 0, 1, 0],
                                        [0, 0,  1]])#0
        
        tdgs_coordinate_info[1,:,:]=np.array([ [1, 0, 0],
                                        [ 0, 1, 0],
                                        [0, 0,  1]])#1
        
        tdgs_coordinate_info[2,:,:]=np.array([ [1, 0, 0],
                                        [ 0, 1, 0],
                                        [0, 0,  1]])#2
        
        tdgs_coordinate_info[3,:,:]=np.array([ [0, 1, 0],
                                        [ 1, 0, 0],
                                        [0, 0,  -1]])#3
        
        tdgs_coordinate_info[4,:,:]=np.array([ [0, 0, 1],
                                        [ 1, 0, 0],
                                        [0, 1,  0]])#4
        
        tdgs_coordinate_info[5,:,:]=np.array([ [0, 0, 1],
                                        [ 1, 0, 0],
                                        [0, 1,  0]])#5
        
        tdgs_coordinate_info[6,:,:]=np.array([ [1, 0, 0],
                                        [ 0, 1, 0],
                                        [0, 0,  1]])#6
        
        

        moved_center_vertices_list_new=apply_cordinate_shift_gt(moved_center_vertices_list,tdgs_coordinate_info,num_linkes)
        for i in range(len(moved_center_vertices_list_new)):
            if i ==2:
                vector_2=np.array([0,0,0.25-0.055])
                vertices=moved_center_vertices_list_new[i]+vector_2
                moved_center_vertices_list_new[i]=vertices
            elif i ==3:
                vector_3=np.array([-0.5,0,-0.4-0.055])
                rotation_fix=np.array([[1,0,0],[0,1,0],[0,0,1]])
                vertices=moved_center_vertices_list_new[i]+vector_3
                vertices=vertices@rotation_fix
                moved_center_vertices_list_new[i]=vertices
            else:
                vertices=moved_center_vertices_list_new[i]
                moved_center_vertices_list_new[i]=vertices

        save_obj(moved_center_vertices_list_new,face_list,color_list,normals_list,args.save_path,args.num_links,file_name_list)

 
if __name__=="__main__":
    main()
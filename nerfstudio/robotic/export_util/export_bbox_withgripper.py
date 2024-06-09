from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
import trimesh
import os

from nerfstudio.robotic.kinematic.uniform_kinematic import *


# scale_factor_gt
# scale_factor_gt = np.array([0.44, 0.40, 0.58])  # cm of the table, length of the table,width of the table, height of the table
scale_factor_gt= np.array([0.148, 0.15, 0.071])  # cm of the base,x,y,z


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

def get_8_corners_from_bbox(bbox,scale_factor):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox

    # time scale_factor to each corner
    xmin, ymin, zmin = xmin / scale_factor[0], ymin / scale_factor[1], zmin / scale_factor[2]
    xmax, ymax, zmax = xmax / scale_factor[0], ymax / scale_factor[1], zmax / scale_factor[2]
    
    corners = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ])

    return corners


def get_bbox_from_8_corners(corners):
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)

    bbox = (min_corner.tolist(), max_corner.tolist())
    return bbox

def convert_pointcloud_to_ply(path):
    vertices_save_list=[]
    faces_save_list=[]
    file_name_list=[]
    bbox_save_list=[]
    bbox_reoriented_save_list=[]
     # vertices, faces
    path_list=os.listdir(path)
    path_list.sort()

    for file in path_list:
        if file.endswith('.ply'):
            # obj_path = os.path.join(path, file)
            ply_path = os.path.join(path, file)
            

            pcd = o3d.io.read_point_cloud(ply_path)
            vertices = np.asarray(pcd.points)
            pcd.estimate_normals()

            # estimate radius for rolling ball
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist   

            raw_mesh, _ = pcd.compute_convex_hull()

            # create the triangular mesh with the vertices and faces from open3d
            tri_mesh = trimesh.Trimesh(np.asarray(raw_mesh.vertices), np.asarray(raw_mesh.triangles),
                                    vertex_normals=np.asarray(raw_mesh.vertex_normals))
            
            
            bbox = tri_mesh.bounding_box.bounds
            bbox_save_list.append(bbox)
            bbox_reorient = tri_mesh.bounding_box_oriented.bounds
            bbox_reoriented_save_list.append(bbox_reorient)
            # vertices = tri_mesh.vertices
            faces = tri_mesh.faces
            # Export the mesh as a PLY file
            # mesh.export(ply_path, file_type='ply')
            
            vertices_save_list.append(vertices)
            faces_save_list.append(faces)
        file_name_list.append(file)
    return vertices_save_list,faces_save_list,file_name_list,bbox_save_list,bbox_reoriented_save_list

def save_list(path,list):
    with open(path, 'w') as f:
        for item in list:
            f.write("%s\n" % item)


def get_bbox_from_base(base_path, scale_factor_gt,original_link_path,original_full_gripper_path,original_gripper_path,output_file_path,experiment_type):
    #load base 
    vertices_save_list,faces_save_list,file_name_list,bbox_save_list,bbox_reoriented_save_list=convert_pointcloud_to_ply(base_path)

    bounding_box_corners = bbox_reoriented_save_list[0]
    
    # this method works for  - + - coordinate system for recenter_vector
    extent_base=bounding_box_corners[1]-bounding_box_corners[0]

    scale=scale_factor_gt/extent_base


    #replace scale by the relative scale_factor
    # scale[0]=scale[0]*1.5

    # scale[1]=scale[1]*1.15
    # scale[2]=scale[2]*1.7

    # Get the coordinates of the lower plane center
    center_vector = bounding_box_corners.mean(axis=0)
    center_vector[2] = bounding_box_corners[0][2]

    output_scale=scale # x is not important, jut fix z 
    recenter_vector=center_vector


    original_scene_list,file_name_list_scene=convert_obj_to_ply_scene(original_link_path)


    num_linkes=8 #7+1
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
   


    movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file_path,experiment_type,scale_factor_pass=output_scale,center_vector_pass=recenter_vector)   

    bbox_save_list=[[]]*num_linkes
    forward_point_list=[[]]*num_linkes
    for i in range(len(original_scene_list)):
        bbox_corners=get_8_corners_from_bbox(original_bbox_nonori_list[i],scale_factor)

        if i==0:
                
            forward_point=bbox_corners+center_vector_gt
            forward_point_list[i]=forward_point
            bbox= get_bbox_from_8_corners(forward_point)
            bbox_save_list[i]=bbox
            # elif i==2:
            #     select_xyz=np.asarray(test_xyz.points)
            #     rotation_inv = inverse_transformation[i-1][:3, :3]
            #     translation_inv = inverse_transformation[i-1][:3, 3]
            #     rotation = final_transformations_list[i-1][:3, :3]
            #     translation = final_transformations_list[i-1][:3, 3]
            #     deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )
            #     forward_point=deform_point
        elif i==7:
            rotation = final_transformations_list_0[i-2][:3, :3]
            translation = final_transformations_list_0[i-2][:3, 3]

            forward_point=bbox_corners
            forward_point=  np.array(forward_point @ rotation.T+ translation )
            forward_point=forward_point+center_vector_gt
            forward_point_list[i]=forward_point
            bbox= get_bbox_from_8_corners(forward_point)
            bbox_save_list[i]=bbox


        else:
            rotation = final_transformations_list_0[i-1][:3, :3]
            translation = final_transformations_list_0[i-1][:3, 3]

            forward_point=bbox_corners
            forward_point=  np.array(forward_point @ rotation.T+ translation )
            forward_point=forward_point+center_vector_gt
            forward_point_list[i]=forward_point
            bbox= get_bbox_from_8_corners(forward_point)
            bbox_save_list[i]=bbox


                

    return bbox_save_list,forward_point_list
if __name__=="__main__":
    # Load the mesh



    # part_path ="/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/link" # grasp 
    part_path ="/home/lou/gs/nerfstudio/exports/splat/no_downscale/gripper_object_dynamic/part/link" # dynamic grasp object
    asset_path= "/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/object"
    output_file_path="/home/lou/gs/nerfstudio/transformation_0416_object_grasp/joint_states_data_0416.txt"

    gripper_path="/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/gripper"
    original_path="/home/lou/Downloads/urdf_2dgs/original_link"

    original_link_path=os.path.join(original_path,"link")
    original_full_gripper_path=os.path.join(original_path,"full_gripper")
    original_gripper_path=os.path.join(original_path,"gripper")
    original_base_path=os.path.join(original_path,"base_gripper_close")
    # original_base_path=os.path.join(original_path,"base_push_bbox")
    # original_base_path=os.path.join(original_path,"base_novel_pose")


    experiment_type = "grasp_object"

    #test this method with grasp_object as well 


    bbox_save_list,forward_point_list=get_bbox_from_base(original_base_path,scale_factor_gt,original_link_path,original_full_gripper_path,original_gripper_path,output_file_path,experiment_type)

    # get bbox from base

    np.savetxt("/home/lou/gs/nerfstudio/exports/splat/no_downscale/gripper_close/bbox_info/bbox_list.txt", np.array(bbox_save_list).reshape(-1, np.array(bbox_save_list).shape[-1]))

    for i in range(len(forward_point_list)):
        forward_point=forward_point_list[i]
        o3d.io.write_point_cloud("/home/lou/Downloads/urdf_2dgs/original_link/test_bbox_forward/"+str(i)+".ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(forward_point)))

    vertices_save_list,faces_save_list,file_name_list,bbox_save_list,bbox_reoriented_save_list=convert_pointcloud_to_ply(part_path)

    vertices_save_list_gripper,faces_save_list_gripper,file_name_list_gripper,bbox_save_list_gripper,bbox_reoriented_save_list_gripper=convert_pointcloud_to_ply(gripper_path)
    object_vertices_save_list,object_faces_save_list,object_file_name_list,object_bbox_save_list,object_bbox_reoriented_save_list=convert_pointcloud_to_ply(asset_path)


    bbox_reoriented_save_list.insert(7,object_bbox_reoriented_save_list[0])
    saved_array=np.array(bbox_reoriented_save_list)
    np.savetxt("/home/lou/gs/nerfstudio/grasp_object_bbox/bbox_list.txt", saved_array.reshape(-1, saved_array.shape[-1]))
    

    # compute scale by the object bounding box

    # ori_table=object_bbox_reoriented_save_list[1] # orientated bbox of 
    # extent_table=ori_table[1]-ori_table[0]
    # # compute center_vector 

    # # scale=extent_table/scale_factor_gt
    # scale=scale_factor_gt/extent_table

    # Extract the corners of the bounding box of link 0
    bounding_box_corners = bbox_reoriented_save_list[0]
    
    # this method works for  - + - coordinate system for recenter_vector
    extent_base=bounding_box_corners[1]-bounding_box_corners[0]

    scale=scale_factor_gt/extent_base

    # scale[1]=scale[1]*-1



    # Get the coordinates of the lower plane center
    center_vector = bounding_box_corners.mean(axis=0)
    center_vector[2] = bounding_box_corners[0][2]
    # experiment_type = "grasp_object"
    


    movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file_path,experiment_type,scale_factor_pass=scale,center_vector_pass=center_vector)   

    individual_transformations_0, final_transformations_list = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=1)  # total length 270



    recenter_matrix = np.eye(4)
    recenter_matrix[:3, 3] = -center_vector_gt
    # final_transformations_list_0.insert(0, recenter_matrix)
    # final_transformations_list.insert(0, recenter_matrix)

    inverse_transformation=inverse_affine_transformation(final_transformations_list_0)


    # test_xyz=o3d.io.read_point_cloud("/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/test_inverse/deform_point2_deform.ply")
    for i in range(len(vertices_save_list)):
            select_xyz=vertices_save_list[i]

            if i==0:
                forward_point=  np.array(select_xyz -center_vector_gt)
                forward_point=forward_point+center_vector_gt
            # elif i==2:
            #     select_xyz=np.asarray(test_xyz.points)
            #     rotation_inv = inverse_transformation[i-1][:3, :3]
            #     translation_inv = inverse_transformation[i-1][:3, 3]
            #     rotation = final_transformations_list[i-1][:3, :3]
            #     translation = final_transformations_list[i-1][:3, 3]
            #     deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )
            #     forward_point=deform_point
            else:
                rotation_inv = inverse_transformation[i-1][:3, :3]
                translation_inv = inverse_transformation[i-1][:3, 3]
                rotation = final_transformations_list[i-1][:3, :3]
                translation = final_transformations_list[i-1][:3, 3]
                
                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )
                forward_point=deform_point
                forward_point=  np.array(deform_point @ rotation.T+ translation )
                forward_point=forward_point+center_vector_gt
                

            o3d.io.write_point_cloud("/home/lou/Downloads/gripper_movement/gripper_part_asset/splat_operation_obj/recenter_link/"+file_name_list[i], o3d.geometry.PointCloud(o3d.utility.Vector3dVector(forward_point)))
        # forward_point=  np.array(deform_point @ rotation.T+ translation )
        # forward_point=forward_point+center_vector_gt



    print("center_vector_gt",center_vector_gt)
    print("scale_factor",scale_factor)



import torch
import numpy as np
import pypose as pp
import os
import cv2
import open3d as o3d
import trimesh



import argparse
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.export_util.urdf_utils.urdf_config import *

from nerfstudio.robotic.kinematic.gripper_utils import reflect_x_axis,reflect_y_axis,reflect_z_axis

from nerfstudio.robotic.kinematic.control_helper import *

from nerfstudio.robotic.export_util.urdf_utils.urdf_helper import *
from nerfstudio.robotic.config.raw_config import export_urdf_to_omnisim_config

# run command: python nerfstudio/robotic/export_util/urdf_utils/urdf.py --part_path ./dataset/roboarm2/urdf/2dgs/arm --save_path ./dataset/roboarm2/urdf/2dgs/recenter_mesh --kinematic_info_path ./dataset/roboarm2/urdf/2dgs/kinematic/kinematic_info.yaml --experiment_type cr3 --scale_factor_gt 1.0 --num_links 7


def main():
    parser = argparse.ArgumentParser(description="export bbox list information from ply file")
    
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

    num_linkes=7

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
    
    center_move_list=computer_center_move(original_bbox_nonori_list,num_linkes)
    





    points_list,face_list,color_list,normals_list,file_name_list,bbox_save_list=convert_obj_to_ply(args.part_path,args.num_links)
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
    # Load the kinematic information


    # only produce urdf for viewer
    # Urdfinfo=Urdfconfig()

    # add parameters to Urdfinfo config file
    # add necessary parameters to Urdfinfo for issac gym in omnisim
    Urdfinfo=export_urdf_to_omnisim_config()
    Urdfinfo.setup_params(args.kinematic_info_path)

    use_kinematic=False
    use_recenter=True
    use_backward=False
    if use_kinematic:
        a,alpha,d,joint_angles_degrees=Urdfinfo.a,Urdfinfo.alpha,Urdfinfo.d,Urdfinfo.joint_angles_degrees
        scale_factor_base=extent_base/np.array(Urdfinfo.base_gt_scale,dtype=np.float32)
        scale_factor=scale_factor_base

        scale_a=np.array([1,1,1,1,1,1])/scale_factor[2]  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
        scale_d=np.array([1,1,1,1,1,1])/scale_factor[2] #  s1 is x axis for d1 ,s4 and s6  is y axis 



        scale_d[3]=1/scale_factor[1]
        scale_d[5]=1/scale_factor[1]

        a=a*scale_a
        d=d*scale_d
        initial_position=np.zeros_like(joint_angles_degrees)
        # apply center_vector_compute to all points
        for i in range(args.num_links):
            points_list[i] = recenter_basedon_kinematic(points_list[i], center_matrix)


        transformations, final_transformations_list=calculate_transformations_mdh_urdf(initial_position,joint_angles_degrees, a, alpha, d,i=0,add_gripper=Urdfinfo.add_gripper,flip_x_coordinate=Urdfinfo.flip_x_coordinate,flip_y_coordinate=Urdfinfo.flip_y_coordinate,flip_z_coordinate=Urdfinfo.flip_z_coordinate)
        # recenter of each part based on edited kinematic
        remap_matrix=inverse_affine_transformation(final_transformations_list)

        # add a identity matrix for the base
        remap_matrix.insert(0,np.eye(4))
        # no gripper and camera this version
        for i in range(args.num_links):
            points_list[i] = remap_basedon_kinematic(points_list[i], remap_matrix[i])

        # second_position=np.ones_like(joint_angles_degrees)*0.1

        # transformations, final_transformations_list_2=calculate_transformations_mdh_urdf(second_position,joint_angles_degrees, a, alpha, d,i=0,add_gripper=Urdfinfo.add_gripper,flip_x_coordinate=Urdfinfo.flip_x_coordinate,flip_y_coordinate=Urdfinfo.flip_y_coordinate,flip_z_coordinate=Urdfinfo.flip_z_coordinate)
        # final_transformations_list_2.insert(0, np.eye(4))
        # for i in range(args.num_links):
        #     points_list[i] = remap_basedon_kinematic(points_list[i], final_transformations_list_2[i])

        save_obj(points_list,face_list,color_list,normals_list,args.save_path,args.num_links,file_name_list)
    elif use_recenter:
        I=np.identity(3)

        gl23d=np.array([[0, 1, 0],
                        [ 0, 0, 1],
                        [1, 0,  0]])
        coordinate_info=np.concatenate([[gl23d]] * num_linkes, axis=0)
        




        coordinate_info[0,:,:]=np.array([ [-1, 0, 0],
                                        [ 0, 0, 1],
                                        [0, 1,  0]])@coordinate_info[0,:,:]#0

        # this link may have issue
        coordinate_info[1,:,:]=np.array([ [0, -1, 0],
            [0, 0, -1],
            [-1, 0, 0]
            ]
            )@coordinate_info[1,:,:]#1


        coordinate_info[2,:,:]=coordinate_info[2,:,:]@np.array([ [0, 1, 0],    
                                        [ 1, 0, 0],
                                        [0, 0,  -1]])#2
        


        # coordinate_info[3,:,:]=np.array([[1, 0, 0],   
        #                                 [ 0, -1, 0],
        #                                 [0, 0, 1]])@coordinate_info[3,:,:]@ rotation_matrix_y@np.array([[0,1,0],
        #                                                                                                 [0,0,1],
        #                                                                                                 [1,0,0]])#3
        coordinate_info[3,:,:]=np.array([[1, 0, 0],   
                                        [ 0, -1, 0],
                                        [0, 0, 1]])@coordinate_info[3,:,:]#3

        coordinate_info[4,:,:]=np.array([[0,-1,0],
                                        [0,0,1],
                                        [1,0,0]])@coordinate_info[4,:,:]#4
        
        coordinate_info[5,:,:]=np.array([[0,1,0],
                                        [0,0,-1],
                                        [1,0,0]])@coordinate_info[5,:,:] #5
        

        coordinate_info[6,:,:]=np.array([[1, 0, 0], 
                                        [ 0, 0, -1],
                                        [0, 1,  0]])@coordinate_info[6,:,:] #
        



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
                    # new_center_list[i][0]=raw[0]-center_move_list[i][0]
                    # new_center_list[i][1]=raw[1]-center_move_list[i][2]
                    # new_center_list[i][2]=raw[2]-center_move_list[i][1]
                    new_center_list[i][0]=0-0.6395
                    new_center_list[i][1]=0
                    new_center_list[i][2]=0+0.25
                if i==3:
                    new_center_list[i]=np.zeros(3)
                    # new_center_list[i][0]=raw[0]+center_move_list[i][0]
                    # new_center_list[i][1]=raw[1]+center_move_list[i][2]
                    # new_center_list[i][2]=raw[2]+center_move_list[i][1]
                    new_center_list[i][0]=0
                    new_center_list[i][1]=0 # by scale computation, write automatic in future 
                    new_center_list[i][2]=0-0.495

            # new_center_list[i]=raw




        
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
        # R_raw2ori_list_recenter,T_raw2ori_list_recenter,R_ori2raw_list_recenter,T_ori2raw_list_recenter=apply_transformation_recenterd(urdf_mesh_vertices_list_reorigt,urdf_mesh_faces_list,raw_mesh_vertices_list,raw_mesh_faces_list,num_linkes)
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

        # save_txt(R_raw2ori_list_recenter,T_raw2ori_list_recenter,R_ori2raw_list_recenter,T_ori2raw_list_recenter, R_raw2ori_list,T_raw2ori_list,R_ori2raw_list,T_ori2raw_list,gs_bbox_list,Matrix_reorientated_list, file_name_list, os.path.join(save_path,'Recenter_info.txt'))


        save_obj(moved_center_vertices_list_new,face_list,color_list,normals_list,args.save_path,args.num_links,file_name_list)

    elif use_backward:
        # recenter the part mesh based on the bounding box center
        for i in range(args.num_links):
            points_list[i] = recenter_basedon_kinematic(points_list[i], center_matrix)
        save_obj(points_list,face_list,color_list,normals_list,args.save_path,args.num_links,file_name_list)
    else:
        for i in range(args.num_links):
            points_list[i] = recenter_basedon_kinematic(points_list[i], center_matrix)
        save_obj(points_list,face_list,color_list,normals_list,args.save_path,args.num_links,file_name_list)

if __name__=="__main__":
    main()
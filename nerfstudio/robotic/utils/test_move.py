from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
# import trimesh
import os






from nerfstudio.robotic.utils import load_transformation_package, relative_tf_to_global,load_txt
# 
# test the movement with new pipeline, with each gs, we first recenter to the origin, then * k_rela_t to make it to the relative position, then *recenter_inv to make it back to original position  (add scale to the k_rela_t)



import rosbag
import rospy

def save_joint_states_to_txt(bag_file, output_file):
    with rosbag.Bag(bag_file, 'r') as bag:
        with open(output_file, 'w') as txt_file:
            for topic, msg, t in bag.read_messages(topics=['/joint_states']):
                # Assuming the joint_states message has an attribute named 'name' and 'position'
                joint_names = msg.name
                joint_positions = msg.position
                # Write joint_states data to txt file
                txt_file.write("Time: {}\n".format(t))
                txt_file.write("Joint Names: {}\n".format(joint_names))
                txt_file.write("Joint Positions: {}\n\n".format(joint_positions))







if __name__ == "__main__":


    # use point cloud to perform mapping

    # part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/original_ply' # 
    part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/part' # novel pose  
    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/recenter_mesh'
    part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/original_link/original_link' #gt


    # bag_file = "/home/lou/gs/nerfstudio/transformation_novelpose/novel_pose.bag"
    # # Specify the output txt file name
    # output_file = "/home/lou/gs/nerfstudio/transformation_novelpose/joint_states_data.txt"
    # # Call the function to save joint_states data to txt
    # save_joint_states_to_txt(bag_file, output_file)


    # load_recenter_info_path ='/home/lou/gs/nerfstudio/Recenter_info.txt'
    # recenter_info=load_txt(load_recenter_info_path)
        # T_final = T_transformed
    I=np.identity(4)
    # recenter_matrix=np.concatenate([[I]] * len(recenter_info), axis=0)
        
    # for i in range(len(recenter_info)):
    #         # replace the bboxes with the new scenes bboxs
    #         if i ==0:
    #             # base in not important
    #             key_info=f'link{i}.ply'
    #             # bboxes[i]=recenter_info[key_info]['gs_bbox_list']
    #         else:


    #             # raw 2 ori recenter  
    #             key_info=f'link{i}.ply'
    #             R_temp=np.zeros((3,3))
    #             T_temp=np.zeros(3)

    #             #raw2ori recenter # not correct
    #             # R_temp[0,:]=recenter_info[key_info][1]
    #             # R_temp[1,:]=recenter_info[key_info][2]
    #             # R_temp[2,:]=recenter_info[key_info][3]
    #             # T_temp[0]=recenter_info[key_info][5]
    #             # T_temp[1]=recenter_info[key_info][6]
    #             # T_temp[2]=recenter_info[key_info][7]
    #             # ori2raw recenter
    #             R_temp[0,:]=recenter_info[key_info][9]
    #             R_temp[1,:]=recenter_info[key_info][10]
    #             R_temp[2,:]=recenter_info[key_info][11]
    #             T_temp[0]=recenter_info[key_info][13]
    #             T_temp[1]=recenter_info[key_info][14]
    #             T_temp[2]=recenter_info[key_info][15]

    #             recenter_matrix[i,:3,:3]=R_temp
    #             recenter_matrix[i,:3,3]=T_temp
        
    transformation_package_path='/home/lou/gs/nerfstudio/transformation_novelpose' 

    # transformation_package_path='/home/lou/gs/nerfstudio/transformation_group1'

    # transformation_package=load_transformation_package(transformation_package_path)


    # rotation_rela, translation_rela,rotation_original,translation_original,rotation_deform,translation_final =relative_tf_to_global(transformation_package)

    # nonzero_timestamp = np.nonzero(translation_rela[4,:,:]) # use link 5 to mark the timestamp

    part_used_path=part_mesh_path # recenter mesh or raw_point
    path_list=os.listdir(part_used_path)
    path_list.sort()

    points_list=[[]]*len(path_list)
    faces_list=[[]]*len(path_list)
    # load the ply file
    i=0
    for file in path_list:
        if file.endswith(".obj"):
        # if file.endswith(".obj"):
            ply_path = os.path.join(part_used_path, file)
            ply_data = o3d.io.read_triangle_mesh(ply_path)
            points=np.asarray(ply_data.vertices)
            faces=np.asarray(ply_data.triangles)
            # ply_data = o3d.io.read_point_cloud(ply_path)
            # points=np.asarray(ply_data.points)
            points_list[i]=points
            faces_list[i]=faces
            i+=1




    
    # scale=0.29

    # recenter the point cloud

    for j in range(len(points_list)):
        link=j+1
        # R_raw2ori=recenter_matrix[link,:3,:3] # recenter contain base
        # T_raw2ori=recenter_matrix[link,:3,3]

        # inverse_R=np.linalg.inv(R_raw2ori)
        link_point=points_list[j]


        line_faces=faces_list[j]
       


    #     # # use this to map it to the orignial position
    #     # new_points=link_point@R_raw2ori.T+T_raw2ori
        
    #     # recentered_point = new_points
    #     # recentered_point=link_point
    #     time=45*10


    #     scale=0.59
    #     # time=nonzero_timestamp[0][20]

    #     rotation_deform_time=rotation_deform[j,time,:,:] #no base

    #     # # need scale compute between original and current scenes for every link and dimension based on the bbox

    #     # scale=3


        # translation_final_time=np.array([-0.2595,0.141,-0.661])  # with out base push_bbox_fix

        # translation_final_time=np.array([-0.25,0.1450,-0.71]) #with base push_bbox_fix


        translation_final_time=np.array([-0.157,0.1715,-0.55]) #with base novel_pose



    #     # rotation_deform_time=rotation_original[j,:,:]
    #     # translation_final_time=translation_original[j,:]*3

    #     #cv2gl matrix

    #     cv2gl=np.array([[1,0,0],[0,0,-1],[0,1,0]])


    #     # rotation_deform_time=np.matmul(cv2gl,rotation_deform_time)
    #     # translation_final_time=np.matmul(translation_final_time,cv2gl.T)

    #     # # # time the k_rela_t and times scale
        deform_point=link_point-translation_final_time
        
            
    #     recentered_point = deform_point
        
    #     # # use the inverse of 
    #     # recentered_point=np.matmul(inverse_R,(deform_point-T_raw2ori).T).T


    #     # save ply
    #     # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/test_deform/deform_point{link}_{time}.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(recentered_point)))
        # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/move2center/deform_point{link}_deform.ply', o3d.geometry.triangle(points=o3d.utility.Vector3dVector(deform_point)))
        o3d.io.write_triangle_mesh(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/move2center/deform_point{link}_deform.obj', o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(line_faces)))
    
    
    # # scale=0.29

    # # recenter the point cloud

    # for j in range(len(points_list)):
    #     link=j+1
    #     # R_raw2ori=recenter_matrix[link,:3,:3] # recenter contain base
    #     # T_raw2ori=recenter_matrix[link,:3,3]

    #     # inverse_R=np.linalg.inv(R_raw2ori)
    #     link_point=points_list[j]



       


    # #     # # use this to map it to the orignial position
    # #     # new_points=link_point@R_raw2ori.T+T_raw2ori
        
    # #     # recentered_point = new_points
    # #     # recentered_point=link_point
    # #     time=45*10


    # #     scale=0.59
    # #     # time=nonzero_timestamp[0][20]

    # #     rotation_deform_time=rotation_deform[j,time,:,:] #no base

    # #     # # need scale compute between original and current scenes for every link and dimension based on the bbox

    # #     # scale=3


    #     # translation_final_time=np.array([-0.2595,0.141,-0.661])  # with out base push_bbox_fix

    #     # translation_final_time=np.array([-0.25,0.1450,-0.71]) #with base push_bbox_fix


    #     translation_final_time=np.array([-0.157,0.1715,-0.55]) #with base novel_pose



    # #     # rotation_deform_time=rotation_original[j,:,:]
    # #     # translation_final_time=translation_original[j,:]*3

    # #     #cv2gl matrix

    # #     cv2gl=np.array([[1,0,0],[0,0,-1],[0,1,0]])


    # #     # rotation_deform_time=np.matmul(cv2gl,rotation_deform_time)
    # #     # translation_final_time=np.matmul(translation_final_time,cv2gl.T)

    # #     # # # time the k_rela_t and times scale
    #     deform_point=link_point-translation_final_time
        
            
    # #     recentered_point = deform_point
        
    # #     # # use the inverse of 
    # #     # recentered_point=np.matmul(inverse_R,(deform_point-T_raw2ori).T).T


    # #     # save ply
    # #     # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/test_deform/deform_point{link}_{time}.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(recentered_point)))
    #     o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/move2center/deform_point{link}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))

    # to map the  xyz_final back to original position



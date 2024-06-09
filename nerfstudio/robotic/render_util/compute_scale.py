from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
import trimesh
import os









from nerfstudio.robotic.utils.utils import load_transformation_package, relative_tf_to_global,load_txt,convert_obj_to_ply_scene






def find_min_max(points):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

            # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    raw_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd,
                    o3d.utility.DoubleVector([radius, radius * 2]))

            # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(raw_mesh.vertices), np.asarray(raw_mesh.triangles),
                                    vertex_normals=np.asarray(raw_mesh.vertex_normals))
    
    tri_mesh.apply_obb()


    min_x = tri_mesh.bounding_box_oriented.bounds[0][0]
    min_y = tri_mesh.bounding_box_oriented.bounds[0][1]
    min_z = tri_mesh.bounding_box_oriented.bounds[0][2]
    max_x = tri_mesh.bounding_box_oriented.bounds[1][0]
    max_y = tri_mesh.bounding_box_oriented.bounds[1][1]
    max_z = tri_mesh.bounding_box_oriented.bounds[1][2]
    return min_x, min_y, min_z, max_x, max_y, max_z




if __name__ == "__main__":


    # use point cloud to perform mapping
    # part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/part'

    # part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/test_deform_part'

    part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/recenter_ply'

    original_load_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/original_link/original_link_nerfstudio'
    part_mesh_path ='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/recenter_mesh_test_deform'


    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/recenter_mesh'
    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/recenter_mesh_test_deform'

    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/recenter_ply'

    
    

    original_scene_list,original_file_name_list=convert_obj_to_ply_scene(original_load_path)

    
    
    original_bbox_list=[]
    original_bbox_nonori_list=[]
    
    
    for original_bbox_asset in original_scene_list:
        
        # original_bbox_ori = original_bbox_asset.bounding_box_oriented.bounds
        original_bbox=original_bbox_asset.bounding_box.bounds
        # original_bbox_list.append(original_bbox_ori)
        original_bbox_nonori_list.append(original_bbox)


    
    load_recenter_info_path ='/home/lou/gs/nerfstudio/Recenter_info.txt'
    recenter_info=load_txt(load_recenter_info_path)
        # T_final = T_transformed
    I=np.identity(4)
    recenter_matrix=np.concatenate([[I]] * len(recenter_info), axis=0)
        
    for i in range(len(recenter_info)):
            # replace the bboxes with the new scenes bboxs
            if i ==0:
                # base in not important
                key_info=f'link{i}.ply'
                # bboxes[i]=recenter_info[key_info]['gs_bbox_list']
            else:


                # raw 2 ori recenter  
                key_info=f'link{i}.ply'
                R_temp=np.zeros((3,3))
                T_temp=np.zeros(3)

                #raw2ori recenter # not correct
                # R_temp[0,:]=recenter_info[key_info][1]
                # R_temp[1,:]=recenter_info[key_info][2]
                # R_temp[2,:]=recenter_info[key_info][3]
                # T_temp[0]=recenter_info[key_info][5]
                # T_temp[1]=recenter_info[key_info][6]
                # T_temp[2]=recenter_info[key_info][7]
                # ori2raw recenter
                R_temp[0,:]=recenter_info[key_info][9]
                R_temp[1,:]=recenter_info[key_info][10]
                R_temp[2,:]=recenter_info[key_info][11]
                T_temp[0]=recenter_info[key_info][13]
                T_temp[1]=recenter_info[key_info][14]
                T_temp[2]=recenter_info[key_info][15]

                recenter_matrix[i,:3,:3]=R_temp
                recenter_matrix[i,:3,3]=T_temp
        
    # transformation_package_path='/home/lou/gs/nerfstudio/transformation_novelpose' 

    transformation_package_path='/home/lou/gs/nerfstudio/transformation_group1'

    transformation_package=load_transformation_package(transformation_package_path)


    rotation_rela, translation_rela,rotation_original,translation_original,rotation_deform,translation_final =relative_tf_to_global(transformation_package)

    nonzero_timestamp = np.nonzero(translation_rela[4,:,:]) # use link 5 to mark the timestamp

    part_used_path=part_mesh_path # recenter mesh 

    # part_used_path=part_ply_path # raw_point
    path_list=os.listdir(part_used_path)
    path_list.sort()

    points_list=[[]]*len(path_list)



    # load the ply file

    i=0
    for file in path_list:
        # if file.endswith(".ply"):
        if file.endswith(".obj"):
            ply_path = os.path.join(part_used_path, file)
            ply_data = o3d.io.read_triangle_mesh(ply_path)
            points=np.asarray(ply_data.vertices)
            # ply_data = o3d.io.read_point_cloud(ply_path)
            # points=np.asarray(ply_data.points)
            points_list[i]=points
            i+=1
    
    # scale=0.29

    I3=np.array((1,1,1))
    scene_scale= np.concatenate([[I3]] * len(recenter_info), axis=0)  #x,y,z


    # recenter the point cloud

    for j in range(len(points_list)):
        link=j+1
        R_raw2ori=recenter_matrix[link,:3,:3] # recenter contain base
        T_raw2ori=recenter_matrix[link,:3,3]

        inverse_R=np.linalg.inv(R_raw2ori)
        link_point=points_list[j]

        min_x, min_y, min_z, max_x, max_y, max_z=find_min_max(link_point)




        original_bbox_i=original_bbox_nonori_list[j]

        scale_x=(max_x+abs(min_x))/(original_bbox_i[1][0]+abs(original_bbox_i[0][0]))
        scale_y=(max_y+abs(min_y))/(original_bbox_i[1][1]+abs(original_bbox_i[0][1]))
        scale_z=(max_z+abs(min_z))/(original_bbox_i[1][2]+abs(original_bbox_i[0][2]))



        # #sugar case
        # scale_x=3
        # scale_y=3
        # scale_z=3

        scene_scale[i]=np.array([scale_x,scale_y,scale_z])
        # # use this to map it to the orignial position
        # new_points=link_point@R_raw2ori.T+T_raw2ori
        

        # time=45*10


        # scale=0.59
        time=nonzero_timestamp[0][20]

        rotation_deform_time=rotation_deform[j,time,:,:] #no base

        # # need scale compute between original and current scenes for every link and dimension based on the bbox

        # scale=3

        translation_final_time=np.zeros_like(translation_final[j,time,:])


        translation_final_time[0]=translation_final[j,time,:][0]*scale_x
        translation_final_time[1]=translation_final[j,time,:][1]*scale_y
        translation_final_time[2]=translation_final[j,time,:][2]*scale_z




        translation_rela_test=np.zeros_like(translation_rela[j,time,:])
        

        translation_rela_test[0]=translation_rela[j,time,:][0]*scale_x
        translation_rela_test[1]=translation_rela[j,time,:][1]*scale_y
        translation_rela_test[2]=translation_rela[j,time,:][2]*scale_z


        translation_original_test=np.zeros_like(translation_original[j,:])

        translation_original_test[0]=translation_original[j,:][0]*scale_x
        translation_original_test[1]=translation_original[j,:][1]*scale_y
        translation_original_test[2]=translation_original[j,:][2]*scale_z



        #cv2gl matrix

        cv2gl=np.array([[1,0,0],[0,0,-1],[0,1,0]])


        # rotation_deform_time=np.matmul(cv2gl,rotation_deform_time)
        # translation_final_time=np.matmul(translation_final_time,cv2gl.T)

        # # # time the k_rela_t and times scale

        deform_point=link_point@rotation_deform_time.T+translation_final_time

        # deform_point=link_point@rotation_deform_time.T+translation_original_test    

        # deform_point=link_point@rotation_rela[j,time,:,:].T+translation_rela_test
        
            
        recentered_point = deform_point



        # save ply
        # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/test_deform/deform_point{link}_{time}.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(recentered_point)))
        # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/test_deform_recenter/deform_point{link}_{time}.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(recentered_point)))


        o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/test_deform/deform_point{link}_{time}.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(recentered_point)))

    # to map the  xyz_final back to original position



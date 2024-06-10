
import numpy as np
import os
import trimesh

def compute_recenter_vector(joint_0_bbox):
    """
    Compute the recenter vector for the given joint vertices.

    Args:
    joint_0_vertices: List of vertices for joint 0.

    Returns:
    recenter_vector: A numpy array representing the recenter vector.
    """
    # Compute the centroid of the vertices
    
    corners=bbox_corners(joint_0_bbox[0],joint_0_bbox[1],joint_0_bbox[2],joint_0_bbox[3],joint_0_bbox[4],joint_0_bbox[5])

    

    # bottom surface mean
    centroid = np.mean(corners[:4], axis=0)
    
    # Compute the recenter vector
    recenter_vector = -centroid
    
    return recenter_vector



def radians_to_degrees(radians):
    return radians * 180 / np.pi

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
            # ply_path = os.path.join(path, file.replace('.obj', '.ply'))
            

            # Load the OBJ file
            mesh = trimesh.load(obj_path)
            mesh.apply_obb()
            # vertices = mesh.vertices
            # faces = mesh.faces
            # # Export the mesh as a PLY file
            # # mesh.export(ply_path, file_type='ply')
            # vertices_save_list.append(vertices)
            # faces_save_list.append(faces)
            scene_list.append(mesh)
        file_name_list_scene.append(file)
    return scene_list,file_name_list_scene

def load_joint_angle(path):
    joint_angle_list=[]
    path_list=os.listdir(path)
    path_list.sort()
    for file in path_list:
        if file.endswith('.npy'):
            joint_angle_path = os.path.join(path, file)
            joint_angle = np.load(joint_angle_path)
            joint_angle_list.append(joint_angle)
    return joint_angle_list








# just for debug
if __name__ == "__main__":



    # # Specify your bag file name
    # bag_file = "group2.bag"
    # # Specify the output txt file name
    # output_file = "joint_states_data.txt"
    # # Call the function to save joint_states data to txt
    # save_joint_states_to_txt(bag_file, output_file)

    
    output_file = "/home/lou/gs/nerfstudio/transformation_novelpose/joint_states_data.txt"


    # output_file = "/home/lou/gs/nerfstudio/transformation_novelpose/joint_states_data_push.txt"
    # output_file = "/home/lou/gs/nerfstudio/transformation_group1/joint_states_data_push.txt"

    # translation_recenter=np.array([-0.157,0.1715,-0.55]) #with base novel_pose



    # center_vector_gt=np.array([-0.25,0.145,-0.71]) #with base group1_bbox_fix push case

    center_vector_gt=np.array([-0.157,0.1715,-0.55]) #with base urdf case  original coordinate x,y,z  -1.42
            # this set works for the bbox_fix group(push case)



    # Read the pre timestamp angle state from txt file
    movement_angle_state = read_txt_file(output_file) # radians

    # Define the transformation matrix at time_0
    T0_i_t0 = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    
    # Define the transformation matrix at time_1
    T0_i_t1 = np.array([[0, 1, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    
    # Compute the transformation
    # Tt0_t1 = transformation_from_t0_to_t1(T0_i_t0, T0_i_t1)
    
    # print(f"Transformation matrix from time_0 to time_1 for joint i:\n{Tt0_t1}")

    # joint_angle_path=''
    # joint_angle=load_joint_angle(joint_angle_path)  # with timestamp


    # test recenter vector  on gs

    
    # part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/original_ply'
    
    
    
    # part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/part' # novel pose  

    part_ply_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/original_ply' # push case
    original_load_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/original_link/original_link_nerfstudio'
    part_mesh_path ='/home/lou/Downloads/2dgsmesh/body'


    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/novel_pose_dynamic/recenter_mesh'
    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/recenter_mesh_test_deform'

    # part_mesh_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/recenter_ply'

    
    
    #original urdf
    original_scene_list,original_file_name_list=convert_obj_to_ply_scene(original_load_path)

    
    
    original_bbox_list=[]
    original_bbox_nonori_list=[]
    
    
    for original_bbox_asset in original_scene_list:
        
        # original_bbox_ori = original_bbox_asset.bounding_box_oriented.bounds
        original_bbox=original_bbox_asset.bounding_box.bounds
        # original_bbox_list.append(original_bbox_ori)
        original_bbox_nonori_list.append(original_bbox)




    # Define your specific joint angles, a, alpha, and d here
    # For example:
    joint_angles_degrees = [0, 0, 0, 0, 0, 0]  # Update with actual angles from the example

    # for i in range(len(joint_angle)):
    #     joint_angles_degrees[i]=radians_to_degrees(joint_angle[i])


    # a_urdf=[0, -0.425, -0.3922, 0, 0, 0]
    # d_urdf=[0.1625, 0, 0, 0.1333, 0.0997, 0.0996]




    # a_list=load_a_from_bbox(original_bbox_nonori_list)
    a = [0, 0, -0.427, -0.357, 0, 0]

    #alpha should be default from urdf

    alpha = [0, np.pi/2, 0.07, 0, np.pi/2, -np.pi/2]

    # d_list=load_d_from_bbox(original_bbox_nonori_list)

    d = [0.147, 0, 0.025, 0.116, 0.116, 0.105]

    # d = [0.147, 0, 0, 0.141, 0.116, 0.105]

    joint_angles_degrees = [0, -np.pi/2, 0, -np.pi/2, 0, 0]  # Update with actual angles from the example

    # use original urdf to compute a,d,scale 

    # x 1.907; y 1.546; z 1.71 for novel pose nerfstudio scale, from ns to real
    scale_a=np.array([1,1,1,1,1,1])/1.29  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
    scale_d=np.array([1,1,1,1,1,1])/1.29 #  s1 is x axis for d1 ,s4 and s6  is y axis 


    # scale_d[0]=1/1.17
    scale_d[3]=1/1.167
    scale_d[5]=1/1.167
    # scale_y=np.array([1,1,1,1,1,1])/1.62
    # scale_z=np.array([1,1,1,1,1,1])/1.62
    # scale=np.zeros((6,3))# test uniform scale for novel pose all joint

    a=a*scale_a
    d=d*scale_d
    

    #use scale to get a,d for the current scenes
    part_used_path=part_mesh_path # recenter mesh or raw_point
    path_list=os.listdir(part_used_path)

    path_list.sort()
    # load point cloud from the part_ply_path, the gs point cloud
    i=0

    points_list=[[]]*len(path_list)
    bbox_list=[[]]*len(path_list)
    face_list=[[]]*len(path_list)
    color_list=[[]]*len(path_list)
    normals_list=[[]]*len(path_list)

    for file in path_list:
        # if file.endswith(".ply"):
        if file.endswith(".obj"):
            ply_path = os.path.join(part_used_path, file)
            ply_data = o3d.io.read_triangle_mesh(ply_path)
            # points=np.asarray(ply_data.vertices)
            # ply_data = o3d.io.read_point_cloud(ply_path)
            points=np.asarray(ply_data.vertices)
            face=np.asarray(ply_data.triangles)
            color=np.asarray(ply_data.vertex_colors)
            normals=np.asarray(ply_data.vertex_normals)
            # mesh.apply_obb()
            points_list[i]=points
            face_list[i]=face
            color_list[i]=color
            normals_list[i]=normals
            # bbox_list[i]=mesh.bounding_box.bounds
            i+=1


    # shift coordinate to right coordinate 

    # center_vector=compute_recenter_vector(bbox_list[0])


    # center_vector_gt=np.array([-0.2595,0.141,-0.661])  #work 


    # center_vector_gt=np.array([-0.157,0.1715,-0.55]) #with base novel_pose

    # extract new a and d from the ns point cloud


    new_a=np.array([0,0,0,0,0,0])
    new_d=np.array([0,0,0,0,0,0])

    # find the point cloud size based on the apply_obb bbox


    # get the new a,d based on the scale


    #apply inverse transformation for recentering the point cloud




    # a should be vertical length and d is the horizontal length

    # Compute the transformations

    

    individual_transformations_0, final_transformations_list_0 = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=0)

    # Print the individual transformations and the final transformation matrix
    for i, t in enumerate(individual_transformations_0, 1):
        print(f"T{i-1}{i}:\n{t}\n")

    print(f"T06 (final transformation from base to end-effector):\n{final_transformations_list_0}")


    # get urdf exportable mesh by the inverse of relative transformation matrix


    time_step=128


    individual_transformations, final_transformations_list = calculate_transformations_mdh(movement_angle_state,joint_angles_degrees, a, alpha, d,i=time_step)

    # apply transformation:



    # test inverse

    inverse_transformation=inverse_affine_transformation(final_transformations_list_0)
    # inverse_transformation=inverse_relative_transformation(final_transformations_list)
    # we apply 

    # test_inverse=True
    test_gripper=False


    if test_gripper:
        pass
    else:
        for i in range(len(path_list)):   
            if i==0:
                link=i
                select_xyz=points_list[i]
                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])


                select_xyz=select_xyz-center_vector_gt
                # no move of the base
                deform_point=  np.array(select_xyz )

                # deform_point=deform_point+center_vector_gt
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)
            elif i==7:
                # gripper
                link=i-1-1
                
                # if test_inverse:
                rotation_inv=inverse_transformation[link][:3,:3]
                translation_inv=inverse_transformation[link][:3,3]
                # else:

                # rotation=final_transformations_list_0[link][:3,:3]
                # translation=final_transformations_list_0[link][:3,3]
                rotation=final_transformations_list[link][:3,:3]
                translation=final_transformations_list[link][:3,3]
                select_xyz=points_list[i]
                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])

                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )

                forward_point=  np.array(deform_point @ rotation.T+ translation )
                forward_point=forward_point+center_vector_gt
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(forward_point)))
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)
            elif i==8:
                # gripper left 0
                link=0
                
                # if test_inverse:
                rotation_inv=inverse_transformation[link][:3,:3]
                translation_inv=inverse_transformation[link][:3,3]
                # else:

                # rotation=final_transformations_list_0[link][:3,:3]
                # translation=final_transformations_list_0[link][:3,3]
                rotation=final_transformations_list[link][:3,:3]
                translation=final_transformations_list[link][:3,3]
                select_xyz=points_list[i]
                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])

                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )

                forward_point=  np.array(deform_point @ rotation.T+ translation )
                forward_point=forward_point+center_vector_gt
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(forward_point)))
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)


            elif i==9:
                # gripper left 1
                link=0
                
                # if test_inverse:
                rotation_inv=inverse_transformation[link][:3,:3]
                translation_inv=inverse_transformation[link][:3,3]
                # else:

                # rotation=final_transformations_list_0[link][:3,:3]
                # translation=final_transformations_list_0[link][:3,3]
                rotation=final_transformations_list[link][:3,:3]
                translation=final_transformations_list[link][:3,3]
                select_xyz=points_list[i]
                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])

                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )

                forward_point=  np.array(deform_point @ rotation.T+ translation )
                forward_point=forward_point+center_vector_gt
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(forward_point)))
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)
            elif i==10:
                # gripper right 0
                link=0
                
                # if test_inverse:
                rotation_inv=inverse_transformation[link][:3,:3]
                translation_inv=inverse_transformation[link][:3,3]
                # else:

                # rotation=final_transformations_list_0[link][:3,:3]
                # translation=final_transformations_list_0[link][:3,3]
                rotation=final_transformations_list[link][:3,:3]
                translation=final_transformations_list[link][:3,3]
                select_xyz=points_list[i]
                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])

                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )

                forward_point=  np.array(deform_point @ rotation.T+ translation )
                forward_point=forward_point+center_vector_gt
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(forward_point)))
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)
            elif i==11:
                # gripper right 1
                link=0
                
                # if test_inverse:
                rotation_inv=inverse_transformation[link][:3,:3]
                translation_inv=inverse_transformation[link][:3,3]
                # else:

                # rotation=final_transformations_list_0[link][:3,:3]
                # translation=final_transformations_list_0[link][:3,3]
                rotation=final_transformations_list[link][:3,:3]
                translation=final_transformations_list[link][:3,3]
                select_xyz=points_list[i]
                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])

                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )

                forward_point=  np.array(deform_point @ rotation.T+ translation )
                forward_point=forward_point+center_vector_gt
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(forward_point)))
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)

            else:
                link=i-1
                
                # if test_inverse:
                rotation_inv=inverse_transformation[link][:3,:3]
                translation_inv=inverse_transformation[link][:3,3]
                # else:
                # rotation=final_transformations_list_0[link][:3,:3]
                # translation=final_transformations_list_0[link][:3,3]
                rotation=final_transformations_list[link][:3,:3]
                translation=final_transformations_list[link][:3,3]
                select_xyz=points_list[i]

                select_faces=np.asarray(face_list[i])

                select_color=np.asarray(color_list[i])

                select_normals=np.asarray(normals_list[i])
                select_xyz=select_xyz-center_vector_gt
                deform_point=  np.array(select_xyz @ rotation_inv.T+ translation_inv )
                
                axis_move_vector=np.zeros((6,3))
                # axis_move_vector[2,:]=np.array([0,0,-0.027])
                deform_point=deform_point+axis_move_vector[link,:]
                forward_point=  np.array(deform_point @ rotation.T+ translation )

                forward_point=forward_point+center_vector_gt
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(forward_point)))
                output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
                output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
                output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
                # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
                o3d.io.write_triangle_mesh(f'/home/lou/Downloads/recenter_2dgs/deform_point{i}_deform.ply',output_mesh)

    #use the push case as the input


    T_time = transformation_from_t0_to_t1(T0_i_t0, T0_i_t1)


    


    # experiment, recenter with sugar parameter (a,d,alpha)




    # apply the T_time on the gs point cloud, test the transformation



    # then apply the recenter vector to the gs point cloud to make it to the original position



    #orignial gs ns, no_receneter




    # has recenter gsns
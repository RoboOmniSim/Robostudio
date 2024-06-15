from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
import trimesh
import os




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

def rigid_transform_3D(A, B):
    # from https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


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


            # 计算连通分量
            triangle_clusters, cluster_n_triangles, cluster_area = obj_data.cluster_connected_triangles()
            
            # 识别最大的连通分量
            largest_cluster_idx = np.argmax(cluster_area)
            triangles_to_remove = [i for i, c in enumerate(triangle_clusters) if c != largest_cluster_idx]
            
            # 移除不属于最大的连通分量的三角形
            obj_data.remove_triangles_by_index(triangles_to_remove)
            
            # 可选地，移除未引用的顶点
            obj_data.remove_unreferenced_vertices()
            
            # 保存清理后的mesh文件

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



def convert_pointcloud_to_ply(path):
    vertices_save_list=[]
    faces_save_list=[]
    file_name_list=[]
     # vertices, faces
    path_list=os.listdir(path)
    path_list.sort()
    for file in path_list:
        if file.endswith('.ply'):
            # obj_path = os.path.join(path, file)
            ply_path = os.path.join(path, file)
            

            pcd = o3d.io.read_point_cloud(ply_path)
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
            


            vertices = tri_mesh.vertices
            faces = tri_mesh.faces
            # Export the mesh as a PLY file
            # mesh.export(ply_path, file_type='ply')
            vertices_save_list.append(vertices)
            faces_save_list.append(faces)
        file_name_list.append(file)
    return vertices_save_list,faces_save_list,file_name_list






def export_urdf_list(urdf_mesh_list,urdf_path):

    for i in range(len(urdf_mesh_list)):
        mesh=trimesh.load(urdf_mesh_list[i])
        mesh.export(urdf_path+str(i)+'.obj', file_type='obj')


def apply_obb(path):
    mesh=trimesh.load(path)
    mesh.apply_obb()
    save_path=path.split('.')[0]+'_obb.obj'
    mesh.export(save_path, file_type='obj')
    return mesh


def compute_bbox(vertices):
     
    bbox = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    return bbox




def apply_transformation(gs_mesh_list_vertices,gs_mesh_list_faces,coordinate_info,num_linkes):
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
        if usepointcloud:
            reshaped_vertices=mesh.vertices.reshape(3,-1)
            reshaped_raw_vertices=raw_mesh.vertices.reshape(3,-1)

            R_raw2ori,T_raw2ori=rigid_transform_3D(reshaped_raw_vertices, reshaped_vertices)

            R_ori2raw,T_ori2raw=rigid_transform_3D(reshaped_vertices, reshaped_raw_vertices)
        else:
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


def apply_transformation_recenterd(move_vertice,move_face,raw_vertice,raw_faces,num_linkes):

    R_raw2ori_list_recenter=[[]]*num_linkes   
    T_raw2ori_list_recenter=[[]]*num_linkes
    R_ori2raw_list_recenter=[[]]*num_linkes
    T_ori2raw_list_recenter=[[]]*num_linkes
    for i in range(num_linkes):
        mesh=trimesh.Trimesh(vertices=move_vertice[i],faces=move_face[i])
        raw_mesh=trimesh.Trimesh(vertices=raw_vertice[i],faces=raw_faces[i])



        # mesh.apply_obb()  # recenter 
        # mesh.export(f'test_{i}.ply', file_type='ply')  #
        if usepointcloud:
            reshaped_vertices=mesh.vertices.reshape(3,-1)
            reshaped_raw_vertices=raw_mesh.vertices.reshape(3,-1)

            R_raw2ori,T_raw2ori=rigid_transform_3D(reshaped_raw_vertices, reshaped_vertices)

            R_ori2raw,T_ori2raw=rigid_transform_3D(reshaped_vertices, reshaped_raw_vertices)
        else:
            R_raw2ori=0
            T_raw2ori=0
            R_ori2raw=0
            T_ori2raw=0
        R_raw2ori_list_recenter[i]=R_raw2ori
        T_raw2ori_list_recenter[i]=T_raw2ori
        R_ori2raw_list_recenter[i]=R_ori2raw
        T_ori2raw_list_recenter[i]=T_ori2raw
        

    return  R_raw2ori_list_recenter,T_raw2ori_list_recenter,R_ori2raw_list_recenter,T_ori2raw_list_recenter

def compute_coordinate_info(gs_bbox_reori_list,original_bbox_list,num_linkes,scale):
    coordinate_info_R=[]*num_linkes
    coordinate_info_t=[]*num_linkes
    for i in range(num_linkes):

        # gs_mesh=trimesh.Trimesh(vertices=gs_mesh_list_vertices[i],faces=gs_mesh_list_faces[i])
        original=original_bbox_list[i]
        bounding_box_oriented = np.array(gs_bbox_reori_list[i]).reshape(3,2)
        # bounding_box_original = compute_bbox(original.vertices)
        bounding_box_original_rescale=scale[i].reshape(3,1)*original.reshape(3,2)
        R,t=rigid_transform_3D(bounding_box_original_rescale,bounding_box_oriented)
        coordinate_info_R.append(R)
        coordinate_info_t.append(t)
    

    return coordinate_info_R,coordinate_info_t

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


def compute_center(bbox,num_linkes):
    center_list=[[]]*num_linkes # 7 is the number of linkes


    # this is the porpotion of the engine center against its bbox
    center_list[0]=np.array([0,0,0])
    center_list[1]=np.array([0,0,0])
    center_list[2]=np.array([0,0,0])
    center_list[3]=np.array([0,0,0])
    center_list[4]=np.array([0,0,0])
    center_list[5]=np.array([0,0,0])
    center_list[6]=np.array([0,0,0])

    
    for i in range(num_linkes):
        upper_box = bbox[i][1] 
        lower_box = bbox[i][0]
        proportion=upper_box/(np.abs(upper_box)+np.abs(lower_box))
        proportion=np.maximum(proportion,0.005)
        center_list[i]=proportion

    return center_list

def computer_center_move(bbox,num_linkes):
    center_move_list=[[]]*num_linkes

    center_move_list[0]=np.array([0,0,0])
    center_move_list[1]=np.array([0,0,0])
    center_move_list[2]=np.array([0,0,0])
    center_move_list[3]=np.array([0,0,0])
    center_move_list[4]=np.array([0,0,0])
    center_move_list[5]=np.array([0,0,0])

    center_move_list[6]=np.array([0,0,0]) # if all 7 links
    
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


def save_txt(R_raw2ori_list_recenter,T_raw2ori_list_recenter,R_ori2raw_list_recenter,T_ori2raw_list_recenter, R_raw2ori_list,T_raw2ori_list,R_ori2raw_list,T_ori2raw_list,gs_bbox_list,Matrix_reorientated_list,file_name_list,path):
    with open(path, 'w') as f:
        for i in range(len(R_raw2ori_list_recenter)):
            file_name=file_name_list[i]
            f.write(f'{file_name}\n')
            f.write('R_raw2ori_list_recenter\n')
            f.write(str(R_raw2ori_list_recenter[i])+'\n')
            f.write('T_raw2ori_list_recenter\n')
            f.write(str(T_raw2ori_list_recenter[i])+'\n')
            f.write('R_ori2raw_list_recenter\n')
            f.write(str(R_ori2raw_list_recenter[i])+'\n')
            f.write('T_ori2raw_list_recenter\n')
            f.write(str(T_ori2raw_list_recenter[i])+'\n')
            f.write('R_raw2ori_list\n')
            f.write(str(R_raw2ori_list[i])+'\n')
            f.write('T_raw2ori_list\n')
            f.write(str(T_raw2ori_list[i])+'\n')
            f.write('R_ori2raw_list\n')
            f.write(str(R_ori2raw_list[i])+'\n')
            f.write('T_ori2raw_list\n')
            f.write(str(T_ori2raw_list[i])+'\n')
            f.write('gs_bbox_list\n')
            f.write(str(gs_bbox_list[i])+'\n')
            f.write('Matrix_reorientated_list\n')
            f.write(str(Matrix_reorientated_list[i])+'\n')

        

def apply_cordinate_shift(urdf_mesh_vertices_list,coordinate_info_R,coordinate_info_t,num_linkes):
    urdf_mesh_vertices_list_reori=[]*num_linkes
    for i in range(num_linkes):
        re=urdf_mesh_vertices_list[i]*coordinate_info_R[i]
        
        urdf_mesh_vertices_list_reori[i]=re
    return urdf_mesh_vertices_list_reori

def apply_cordinate_shift_gt(urdf_mesh_vertices_list,coordinate_info_R,num_linkes):
    urdf_mesh_vertices_list_reori=[[]]*num_linkes
    for i in range(num_linkes):
        re=np.dot(urdf_mesh_vertices_list[i],coordinate_info_R[i])
        
        urdf_mesh_vertices_list_reori[i]=re
    return urdf_mesh_vertices_list_reori

def re_center(urdf_mesh_vertices_list,new_center_list):
    for i in range(len(urdf_mesh_vertices_list)):
        vertices=urdf_mesh_vertices_list[i]+new_center_list[i]
        
        urdf_mesh_vertices_list[i]=vertices
    return urdf_mesh_vertices_list


def pre_process_part(save_path,gs_arm_path):

    obj_data = o3d.io.read_triangle_mesh(gs_arm_path)
    file_name=gs_arm_path.split('/')[-1]

    # 计算连通分量
    triangle_clusters, cluster_n_triangles, cluster_area = obj_data.cluster_connected_triangles()
            
    # 识别最大的连通分量
    largest_cluster_idx = np.argmax(cluster_area)
    triangles_to_remove = [i for i, c in enumerate(triangle_clusters) if c != largest_cluster_idx]
            
    # 移除不属于最大的连通分量的三角形
    obj_data.remove_triangles_by_index(triangles_to_remove)
            
    # 可选地，移除未引用的顶点
    obj_data.remove_unreferenced_vertices()
            
    vertices=np.asarray(obj_data.vertices)
    face=np.asarray(obj_data.triangles)
    color=np.asarray(obj_data.vertex_colors)
    normals=np.asarray(obj_data.vertex_normals)


    arm_mesh=trimesh.Trimesh(vertices=vertices,faces=face)
    arm_mesh.apply_obb()
    re_oriented_vertices=arm_mesh.vertices
    re_oriented_faces=arm_mesh.faces


    deform_point=re_oriented_vertices
    select_faces=re_oriented_faces
    select_color=color
    select_normals=normals

          
    output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
    output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
    output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals)

                
    # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
    o3d.io.write_triangle_mesh(os.path.join(save_path,file_name),output_mesh)

    return re_oriented_vertices,re_oriented_faces
# Example usage
            
if __name__ == '__main__':
    num_linkes=7
    # save_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/recenter_mesh'
    # gs_load_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/urdf/workspace/gs_mesh/linkage'
    save_path = './recenter_2dgs'
    output_dir = './cleaned_meshes'
    gs_arm_path='/home/lou/Downloads/urdf_2dgs/update_version/fuse_unbounded_arm_only.ply'

    # re_oriented_vertices,re_oriented_faces=pre_process_part(save_path,gs_arm_path)
    # sugar nesg
    # gs_load_path='./plyresult'

    # gs_load_path = './2dgsmesh/body'  # 2dgs mesh
    gs_load_path = '/home/lou/Downloads/urdf_2dgs/2dgsmesh/body'  # 2dgs mesh

    # gripper_load_path = './2dgsmesh/gripper'  # 2dgs mesh

    # original_load_path='./workspace/original_link' # sugar

    original_load_path = '/home/lou/Downloads/urdf_2dgs/original_link_nogripper'  # nerfstudio


    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    usemesh=True
    usepointcloud=not usemesh
    if usepointcloud:

        # 7 represent gripper
        gs_mesh_list_vertices,gs_mesh_list_faces,file_name_list=convert_pointcloud_to_ply(gs_load_path)
        # save_obj(gs_mesh_list_vertices,gs_mesh_list_faces,save_path,num_linkes,file_name_list)
    else:
        gs_mesh_list_vertices,gs_mesh_list_faces,color_list,normals_list,file_name_list=convert_obj_to_ply(gs_load_path,num_linkes)
        
    original_scene_list,original_file_name_list=convert_obj_to_ply_scene(original_load_path)







    gs_bbox_list=[]
    original_bbox_list=[]
    original_bbox_nonori_list=[]
    # scale diff between the gs_mesh and the original mesh due to the gs training intrinsic variance 
    scale=[[]]*num_linkes

    for mesh_vertices in gs_mesh_list_vertices:
        
        gs_bbox = compute_bbox(mesh_vertices)
        gs_bbox_list.append(gs_bbox)

    for original_bbox_asset in original_scene_list:
        
        original_bbox_ori = original_bbox_asset.bounding_box_oriented.bounds
        original_bbox=original_bbox_asset.bounding_box.bounds
        original_bbox_list.append(original_bbox_ori)
        original_bbox_nonori_list.append(original_bbox)
    center_list=compute_center(original_bbox_list,num_linkes)

    

    # after have the new_center_list, we can use it to generate the new mesh
    
    center_move_list=computer_center_move(original_bbox_nonori_list,num_linkes)
    
    
    coordinate_info=0

    


    urdf_mesh_vertices_list,urdf_mesh_faces_list,raw_mesh_vertices_list,raw_mesh_faces_list,R_raw2ori_list,T_raw2ori_list,R_ori2raw_list,T_ori2raw_list,Matrix_reorientated_list=apply_transformation(gs_mesh_list_vertices,gs_mesh_list_faces,coordinate_info,num_linkes)


    cv2gl=np.array([[1,0,0],[0,0,-1],[0,1,0]])


    theta = np.pi / 2
    rotation_matrix_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
    ])


    for i in range(num_linkes):
        cv_vertice=urdf_mesh_vertices_list[i]
        # gl_vertice=cv_vertice@cv2gl@rotation_matrix_y
        gl_vertice=cv_vertice@rotation_matrix_y@rotation_matrix_x
        urdf_mesh_vertices_list[i]=gl_vertice
    # save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,color_list,normals_list,save_path,num_linkes,file_name_list)

    gs_bbox_reori_list=[]
    for mesh_vertices in urdf_mesh_vertices_list:
        
        gs_bbox_reori = compute_bbox(mesh_vertices)
        gs_bbox_reori_list.append(gs_bbox_reori)


    for i in range(num_linkes):
        scale[i]=(((np.abs(gs_bbox_reori_list[i][1])+np.abs(gs_bbox_reori_list[i][1])))/2)/((np.abs(original_bbox_list[i][1])+np.abs(original_bbox_list[i][0]))/2)


    coordinate_info_R,coordinate_info_t=compute_coordinate_info(gs_bbox_reori_list,original_bbox_list,num_linkes,scale)

    # urdf_mesh_vertices_list_reori=apply_cordinate_shift(urdf_mesh_vertices_list,coordinate_info_R,coordinate_info_t,num_linkes)

    # save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,save_path,num_linkes,file_name_list)
    # save_obj(urdf_mesh_vertices_list_reori,urdf_mesh_faces_list,save_path,num_linkes)

    # coordinate_info=np.zeros((num_linkes,3,3))

    #ground truth coordinate info

    # nerfstudio coordinate


    #0-5 are link 1-6

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

    urdf_mesh_vertices_list_reorigt=apply_cordinate_shift_gt(urdf_mesh_vertices_list,coordinate_info,num_linkes)

    urdf_bbox_reori_list=[]
    for mesh_vertices in urdf_mesh_vertices_list_reorigt:
        
        urdf_bbox_reori = compute_bbox(mesh_vertices)
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




    
    moved_center_vertices_list=re_center(urdf_mesh_vertices_list_reorigt,new_center_list)




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


    save_obj(moved_center_vertices_list_new,urdf_mesh_faces_list,color_list,normals_list,save_path,num_linkes,file_name_list)

    # save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,save_path,num_linkes,file_name_list)

    # key info, scale difference, then is to reorient the bbox to the xyz axis, then apply the recenter and rescale, then move the center to the new motor engine, finish the computation   



    # the reorient and recenter is very important for the 4d rendering for the difference in information 


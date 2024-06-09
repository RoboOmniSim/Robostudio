import open3d as o3d

import numpy as np
import torch
# from pytorch3d.ops import knn_points



# use the nerf camera dataparser for the uniform or nerfacto sampling 




# then use the knn and barycentric resampling of the gaussian bolbs based on the uniform or nerfacto sampling for the point cloud for mesh reconstruction





# assign the color of point cloud based on guassian bolbs from the barycentric coordinates 



# compute normal based on gcno



# then use possion mesh reconstrucion

# def resample_mesh(mean,scale=0,rotation=0):




#     rotated_vertices=generate_tetrahedron_vertices(mean,scale,rotation)



#     # or train a nerfacto and use the nerfacto sampler
#     points=sample_points(rotated_vertices) 


#     return points




    # reproject the mesh to the sampled point and find the 6 knn


    





    # use barycentric for resampling    

    
    # first construct triangle based on the gaussian bolbs and the rotation and cov of the guassian bolbs



    # Then find knn 15 based on gaussian center ,

    # compute the barycentric for each 15 points, find the inside points

    # then use barycentric to sample 6 out of it 

    # or use depth to compute normal 



    # this will generate rotation and translation vaired point cloud for the mesh reconstruction




    # then compute normal based on gcno or other normal estimation method


 # only deal with the mesh inside the operation bbox  
    
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
    

def generate_tetrahedron_vertices(mean,scale,rotation):
        # Generate vertices for a tetrahedron centered at the mean with a scale based on the Gaussian scale

        rotation_matrix=quaternion_to_matrix(torch.tensor(rotation, dtype=torch.float)).cpu().numpy()

        n=mean.shape[0]
        vertices_list=[]
        for i in range(n):
            scale_i=np.diag(scale[i])
            # or maybe need pre-set scale for the tetrahedron
            base_scale = np.max(np.sqrt(scale_i))  # Use the largest scale dimension for the tetrahedron size
            vertices = np.array([
                [1, 0, -1 / np.sqrt(2)],
                [-1, 0, -1 / np.sqrt(2)],
                [0, 1, 1 / np.sqrt(2)],
                [0, -1, 1 / np.sqrt(2)]
            ]) * base_scale
            vertices_r=vertices@rotation_matrix[i,:,:].T
            vertices_list.append(vertices_r)
            
        
        vertices_reuse=np.array(vertices_list).reshape(-1,3)
        vertices_full=mean.repeat(4,axis=0)
        
        # Rotate and translate the vertices
        rotated_vertices = vertices_full + vertices_reuse  # shape: (n*4, 3)
        return rotated_vertices 


def compute_barycentric(points,rotated_vertices):
      
    bary_coords = []

    for n in range(points.shape[0]):
        # Compute the barycentric coordinates of each point in the tetrahedron
        bary_coords.append(np.linalg.solve(rotated_vertices, points[n]))


    return np.array(bary_coords)





def select_inside_points(bary_coords):
    # Select the points inside the tetrahedron
    inside_points = np.array([bary for bary in bary_coords if np.all(bary >= 0) and np.sum(bary) <= 1])
    return inside_points




def sample_points(rotated_vertices):
    bary=0
    sampled_points=rotated_vertices*bary


    return sampled_points


def reconstruct_mesh(sample_points, guassian_color,bary_coords):
    o3d_pc=o3d.geometry.PointCloud()
    o3d_pc.points=o3d.utility.Vector3dVector(sample_points)
    #compute normal
    o3d_pc.estimate_normals()

    # add color
    mesh_color=guassian_color*bary_coords

    o3d_pc.colors=o3d.utility.Vector3dVector(mesh_color)
    

    # generate mesh from point cloud
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pc, depth=9)

    return mesh




def resample_mesh(means3d,shs, colors_precomp,opacities,scale,rotation,cov3D_precomp,load_path=None,save_path=None):
    """
    
    means3d: 3d means of the gaussian bolbs n*3
    rotation: rotation matrix of the gaussian bolbs n*4 quaternion
    scale: scale of the gaussian bolbs n*3
    opacities: opacities of the gaussian bolbs n*1
    
    return:
    xyz: resampled point cloud (pass this xyz as pcd to the renderer in gaustudio mesh)
    """


    # make every input as numpy array
    means3d=np.array(means3d)
    shs=np.array(shs)
    colors_precomp=np.array(colors_precomp)
    opacities=np.array(opacities)
    scale=np.array(scale)
    rotation=np.array(rotation)
    cov3D_precomp=np.array(cov3D_precomp)



    opacities_mask=opacities>0.1

    # only deal with opacities >0.2


    # to keep the rasterize same dimension
    means3d_clean_opacity=means3d[opacities_mask]
    rotation_clean_opacity=rotation[opacities_mask]
    scale_clean_opacity=scale[opacities_mask]
    opacities_clean_opacity=opacities[opacities_mask]
    cov3D_clean_opacity=cov3D_precomp[opacities_mask]
    colors_clean_opacity=colors_precomp[opacities_mask]
    shs_clean_opacity=shs[opacities_mask]


    # resample points,
    # rotated_vertices=generate_tetrahedron_vertices(means3d_clean_opacity,scale_clean_opacity,rotation_clean_opacity)
    # the operation bbox
    
    x_min=-0.9
    x_max=-0.4
    y_min=-0.3
    y_max=2.1
    z_min=-1.1
    z_max=0.2


    bbox=np.array((x_min, y_min, z_min, x_max, y_max, z_max))


    point_bbox_mask=np.all((means3d_clean_opacity>=bbox[:3]) & (means3d_clean_opacity<=bbox[3:]),axis=1)


    means_3d_clean_bbox=means3d_clean_opacity[point_bbox_mask]
    means_3d_clean_outbbox=means3d_clean_opacity[~point_bbox_mask]
    rotation_clean_bbox=rotation_clean_opacity[point_bbox_mask]
    rotation_clean_outbbox=rotation_clean_opacity[~point_bbox_mask]
    scale_clean_bbox=scale_clean_opacity[point_bbox_mask]
    scale_clean_outbbox=scale_clean_opacity[~point_bbox_mask]

    opacities_clean_bbox=opacities_clean_opacity[point_bbox_mask]
    opacities_clean_outbbox=opacities_clean_opacity[~point_bbox_mask]
    cov3D_clean_bbox=cov3D_clean_opacity[point_bbox_mask]
    cov3D_clean_outbbox=cov3D_clean_opacity[~point_bbox_mask]
    colors_clean_bbox=colors_clean_opacity[point_bbox_mask]
    colors_clean_outbbox=colors_clean_opacity[~point_bbox_mask]
    shs_clean_bbox=shs_clean_opacity[point_bbox_mask]
    shs_clean_outbbox=shs_clean_opacity[~point_bbox_mask]





    rotated_vertices=generate_tetrahedron_vertices(means_3d_clean_bbox,rotation_clean_bbox,scale_clean_bbox)


    # then farthest_point_down_sample by part  defination of part are given by the bounding box of the mesh
    o3d_pc=o3d.geometry.PointCloud()
    o3d_pc.points=o3d.utility.Vector3dVector(rotated_vertices)
    

    # down sample again
    o3d_pc.uniform_down_sample(0.25)


    # maybe add moving least square here

    
    # cleaning use open3d mesh


    # remove_degenerate_triangles()
    # remove_duplicated_triangles()
    # remove_duplicated_vertices()
    # remove_non_manifold_edges()

    # meshlab application for mesh

    # cleaning and repairing 
    # remove isolated pieces
    # remove duplicate faces


                       

    # smoothing
    # laplacian smoothing


    #remeshing
    #apply alpha shape

    xyz=np.asarray(o3d_pc.points)

    # this will need barycentric, but temporarily use uniform sampling

    rot_mesh=rotation_clean_bbox[:xyz.shape[0],:] # test 
    scale_mesh=scale_clean_bbox[:xyz.shape[0],:]
    opacities_mesh=opacities_clean_bbox[:xyz.shape[0],:]
    cov3D_mesh=cov3D_clean_bbox[:xyz.shape[0],:]
    colors_mesh=colors_clean_bbox[:xyz.shape[0],:]
    shs_mesh=shs_clean_bbox[:xyz.shape[0],:]

    
    xyz_out=np.concatenate((means_3d_clean_outbbox,xyz),axis=0)
    rot_mesh_out=np.concatenate((rotation_clean_outbbox,rot_mesh),axis=0)
    scale_mesh_out=np.concatenate((scale_clean_outbbox,scale_mesh),axis=0)
    opacities_mesh_out=np.concatenate((opacities_clean_outbbox,opacities_mesh),axis=0)
    cov3D_mesh_out=np.concatenate((cov3D_clean_outbbox,cov3D_mesh),axis=0)
    colors_mesh_out=np.concatenate((colors_clean_outbbox,colors_mesh),axis=0)
    shs_mesh_out=np.concatenate((shs_clean_outbbox,shs_mesh),axis=0)




    return xyz_out,rot_mesh_out,scale_mesh_out,opacities_mesh_out,cov3D_mesh_out,colors_mesh_out,shs_mesh_out














if __name__ == "__main__":
    mean=np.array([[1,2,3],[4,5,6]])
    scale=np.array([[1,2,3],[4,5,6]])
    rotation=np.array([[1,2,3,4],[5,6,7,8]])
    output=generate_tetrahedron_vertices(mean,scale,rotation)
import torch
import numpy as np
import open3d as o3d
import os



import argparse




def reorient_mesh(input_mesh_path, output_mesh_path,re_orientation_matrix):
    """
    Args:
        input_mesh_path: path to the input mesh
        output_mesh_path: path to the output mesh
        re_orientation_matrix: the reorientation matrix from nerfstudio

    Returns:
        mesh: the reoriented mesh
    
    
    """


    transform_matrix_train=re_orientation_matrix
                
    transform_matrix_train[0:3, 1:3] *= -1
    transform_matrix_train = transform_matrix_train[np.array([0, 2, 1]),:]

    transform_matrix_train_concat=torch.eye(4)
    transform_matrix_train_concat[:3,:]=transform_matrix_train

    mesh = o3d.io.read_triangle_mesh(input_mesh_path)

   

    sample_vertices=mesh.vertices  


    point_3d = torch.tensor(np.asarray(sample_vertices)).float()


    fused_point_cloud = (
                torch.cat(
                    (
                        point_3d,
                        torch.ones_like(point_3d[..., :1]),
                    ),
                    -1,
                )
                @ transform_matrix_train_concat.T
            )


    reseted_vertices = fused_point_cloud[..., :3].numpy()

    mesh.vertices=o3d.utility.Vector3dVector(reseted_vertices)

    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    return mesh




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    
    # Add arguments
    parser.add_argument('--input_mesh_path', type=str, help='path to the input mesh file')
    parser.add_argument('--output_mesh_path', type=str, help='path to save axis-aligned mesh')
    parser.add_argument('--re_orientation_matrix', type=str, help='path to load orientation matrix')
    
    args = parser.parse_args()
    
    
    # '/home/lou/gs/2d-gaussian-splatting/output/cr3/train/ours_30000/fuse_unbounded_post_arm.obj'

    # '/home/lou/gs/2d-gaussian-splatting/output/cr3/train/ours_30000/fuse_unbounded_post_arm_reori.obj'

    # this is the transform matrix from nerfstudio dataparser to rescale and reorient the mesh

    re_orientation_matrix=args.re_orientation_matrix
    re_orientation_matrix = torch.tensor([
                        [ 0.9999,  0.0017, -0.0131, -0.0276],
                        [ 0.0017,  0.9674,  0.2533, -0.1400],
                        [ 0.0131, -0.2533,  0.9673,  0.0363]
    ])
    reorient_mesh(args.input_mesh_path, args.output_mesh_path,re_orientation_matrix)
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple
import open3d as o3d
import  os
import sys
import torch
import torch.nn.functional as F





def get_mesh(self,dataparser,bbox,map_to_tensors,colors) :
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        cameras=dataparser.get_camera()
        gaussian=map_to_tensors['positions']
        semantic_id=map_to_tensors['semantic_id']
        Path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/splat.ply'

        mesh=o3d.io.read_triangle_mesh(Path)
        idx=mesh_gaussian_binding(mesh,gaussian)

        points=resample_mesh(cameras)


        guassian_color=colors[semantic_id]
        bary_coords=0

        coarsemesh=reconstruct_mesh(points, guassian_color,bary_coords)
        decimation_target= 100000
        decimated_o3d_fg_mesh = coarsemesh.simplify_quadric_decimation(decimation_target)




        decimated_o3d_fg_mesh.remove_degenerate_triangles()
        decimated_o3d_fg_mesh.remove_duplicated_triangles()
        decimated_o3d_fg_mesh.remove_duplicated_vertices()
        decimated_o3d_fg_mesh.remove_non_manifold_edges()



        o3d.io.write_triangle_mesh("decimated_mesh.ply", decimated_o3d_fg_mesh)
        return idx
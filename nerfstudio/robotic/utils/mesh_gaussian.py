import torch
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

# import pytorch3d



def k_nearest_sklearn(self, x: torch.Tensor, k: int):
    """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
    """
        # Convert tensor to numpy array
    x_np = x.cpu().numpy()

        # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)




def mesh_gaussian_binding(mesh,gaussian,K=1):

    mesh = o3d.io.read_triangle_mesh(mesh)

    vertices = np.asarray(mesh.vertices)

    vertices = torch.tensor(vertices).float().cuda()


    xyz=gaussian # guassian center

    xyz = torch.tensor(xyz).float().cuda()

    
    # closest_points_idx = pytorch3d.ops.ball_query(xyz[None], vertices[None], K=K).idx[0]
    # closest_points_idx=ball_query(xyz,vertices,K=K)
    closest_points_idx=0
    return closest_points_idx








import torch
import pypose
import numpy as np
import math
# Extrinsic matrix

from plyfile import PlyData, PlyElement



# our projection code is written in gsplat, but original gs has different order of xyz by glm difference with colmap

# so we setup the extrinsic matrix by apply transformation and the projected pixel should be the same anyway 



# from gsplat.sh import num_sh_bases, spherical_harmonics
def load_ply( path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, features_dc, features_extra, opacities, scales, rots



def transformPoint4x4(points, matrix):
    """
    Transforms 3D points using a 4x4 projection matrix.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing 3D points.
        matrix (torch.Tensor): Tensor of shape (3, 4) representing the projection matrix.

    Returns:
        torch.Tensor: Transformed points of shape (N, 4).
    """

    transformed = torch.matmul(points, matrix)
      # (N, 4)
    return transformed


def unproject_points(u, v, depth, projmatrix, res):
    """
    Unprojects 2D pixel coordinates to 3D world coordinates using a camera-to-world projection matrix.

    Args:
        u (torch.Tensor): Tensor of shape (N,) containing the x-coordinates of the pixel positions.
        v (torch.Tensor): Tensor of shape (N,) containing the y-coordinates of the pixel positions.
        depth (torch.Tensor): Tensor of shape (N,) containing the depth values at each pixel.
        projmatrix (torch.Tensor): Tensor of shape (4, 4) representing the camera-to-world projection matrix.
        res (tuple): Tuple (height, width) representing the image resolution.

    Returns:
        world_points (torch.Tensor): Tensor of shape (N, 3) containing the 3D world coordinates.
    """
    # Convert pixel coordinates to Normalized Device Coordinates (NDC)
    height, width = res
    x_ndc = (u / (width - 1)) * 2 - 1  # Scale to [-1, 1]
    y_ndc = 1 - (v / (height - 1)) * 2  # Scale to [-1, 1] and flip y-axis

    # Stack NDC coordinates and depth to create clip space coordinates
    # Assuming depth is in the same space as NDC z-coordinate (between -1 and 1)
    # If depth is in view space (positive values), you'll need to map it to NDC space using projection parameters
    ones = torch.ones_like(x_ndc)
    clip_coords = torch.stack([x_ndc, y_ndc, depth, ones], dim=-1)  # (N, 4)

    # Transform clip coordinates to world coordinates
    # Use the camera-to-world projection matrix
    projmatrix_inv = torch.inverse(projmatrix)
    world_coords_hom = clip_coords @ projmatrix_inv.T  # (N, 4)

    # Perform perspective divide
    epsilon = 1e-7
    w = world_coords_hom[:, 3] + epsilon
    world_coords = world_coords_hom[:, :3] / w.unsqueeze(1)

    return world_coords

def project_points(sample_points, projmatrix, res):
    """
    Projects 3D points to 2D pixel coordinates using a projection matrix.

    Args:
        sample_points (torch.Tensor): Tensor of shape (N, 3) containing 3D points.
        projmatrix (torch.Tensor): Tensor of shape (4, 4) representing the projection matrix.
        res (tuple): Tuple (height, width) representing the image resolution.

    Returns:
        u (torch.Tensor): Tensor of shape (N,) containing the x-coordinates of the pixel positions.
        v (torch.Tensor): Tensor of shape (N,) containing the y-coordinates of the pixel positions.
    """
    # Transform points using the projection matrix
    p_hom = transformPoint4x4(sample_points, projmatrix)  # (N, 4)

    # Perform perspective divide
    epsilon = 1e-7
    p_w = 1.0 / (p_hom[:, 3] + epsilon)  # (N,)
    p_proj = p_hom[:, :3] * p_w.unsqueeze(1)  # (N, 3)

    # Map from NDC to screen coordinates
    # Assuming p_proj is in NDC space [-1, 1], map to pixel coordinates
    height, width = res
    u = (p_proj[:, 0] * 0.5 + 0.5) * (width - 1)
    v = (1.0 - (p_proj[:, 1] * 0.5 + 0.5)) * (height - 1)  # Flip y-axis

    # Clamp pixel coordinates to be within image bounds
    u = u.clamp(0, width - 1)
    v = v.clamp(0, height - 1)

    return u, v

# Example usage:

# Sample points
sample_points = torch.tensor([[1.0492, 1.1724, 1.1444],
        [1.0022, 1.1291, 0.9563],
        [1.1931, 0.8935, 0.9443]], device='cuda:0', requires_grad=True)

# Define your intrinsic matrix (3x3)
intrinsic = torch.tensor([
    [2.1017e+03, 0.0000e+00, 9.6000e+02,0],
    [0.0000e+00, 2.1969e+03, 5.4000e+02,0],
    [0.0000e+00, 0.0000e+00, 1.0000e+00,0]
], device='cuda:0')

# Convert intrinsic matrix to 4x4


# Your extrinsic matrix (4x4)
extrinsic = torch.tensor([
    [1, 0, 0, -1],
    [0, 0, 1, 1.1],
    [0,-1, 0, 1],
    [0, 0, 0, 1]
], device='cuda:0')



xyz, features_dc, features_extra, opacities, scales, rots=load_ply(path='/home/lou/physics/DreamPhysics/data_force/object.ply')

xyz=torch.tensor(xyz, device='cuda:0')
features_dc=torch.tensor(features_dc, device='cuda:0')
features_extra=torch.tensor(features_extra, device='cuda:0')
opacities=torch.tensor(opacities, device='cuda:0')
scales=torch.tensor(scales, device='cuda:0')
rots=torch.tensor(rots, device='cuda:0')



colors_crop = torch.cat((features_dc, features_extra), dim=2)


colors_crop=colors_crop.permute(0, 2, 1)
viewdirs = xyz- extrinsic[..., :3, 3]  # (N, 3)

viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
n =3
# rgbs = spherical_harmonics(n, viewdirs, colors_crop)
# rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

# rgbs=rgbs.to('cpu').detach().numpy()
xyz=xyz
# If incorporating znear and zfar
def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    P_persp = torch.zeros((4, 4), dtype=torch.float32, device='cuda:0')

    P_persp[0, 0] = 1 / tanHalfFovX
    P_persp[1, 1] = 1 / tanHalfFovY
    P_persp[2, 2] = -(zfar + znear) / (zfar - znear)
    P_persp[2, 3] = -2 * zfar * znear / (zfar - znear)
    P_persp[3, 2] = -1

    return P_persp

znear=0.1
zfar=100
FoVx_rad=0.856
FoVy_rad=0.48
P_persp = get_projection_matrix(znear, zfar, FoVx_rad, FoVy_rad)

# Full projection transform
full_proj_transform = P_persp @ extrinsic


# gs to gsplat trasformation
full_proj_transform[0:3, 1:3] *= -1
full_proj_transform = full_proj_transform[np.array([0, 2, 1]),:]



# Image resolution
res = (1080, 1920)

# Project the points
u, v = project_points(xyz, full_proj_transform, res)

# Print the pixel coordinates
print("Pixel coordinates (u, v):")
# for i in range(100):
#     print(f"Point {i}: ({u[i].item()}, {v[i].item()})")

j=0
uin_range = (u >= 100) & (u < 1900)
vin_range = (v >= 100) & (v < 1070)
for i in range(len(u)):

    if uin_range[i] and vin_range[i]:
      j+=1

print(j)


# optical flow utils

# reproject all gs to pixel( same shape as gs)


# load the mask or optical flow saved in the pixel to the gs


optical_flow = torch.randn(1080, 1920, 2).to('cuda')  # e.g., a flow map with 2 channels for (x, y) flow


pixel_coords= torch.stack((u, v), dim=1).long()
# Ensure pixel_coords are within image bounds
pixel_coords[:, 0].clamp_(0, optical_flow.shape[1] - 1)  # Width dimension
pixel_coords[:, 1].clamp_(0, optical_flow.shape[0] - 1)  # Height dimension

# Sample optical flow values at each pixel corresponding to gs
flow_at_gs = optical_flow[pixel_coords[:, 1], pixel_coords[:, 0]] 

print(flow_at_gs.shape)  # (N, 2) where N is the number of gs points



# the depth is actually the difference between t1 and t0 value from depth flow
depth=torch.tensor([1.0, 1.0, 1.0], device='cuda:0')  

# the idea is that we have a set of gs bind to this pixel from flow_at_gs , which means we can trace the value of flow_at_gs 
# and find its corresponding initail pixel position from the pixel_coords

# this generate the pixel difference


# then we can trace for the depth value from t0 and t1 to get the depth value diffence on this gs index

# this gives the 3d trajectory of the gs from t0 to t1 

# the unproject point is not necessary under this setting


projmatrix=full_proj_transform

unproject_points(u, v, depth, projmatrix, res)
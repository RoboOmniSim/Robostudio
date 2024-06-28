from typing import Optional
import torch
import kornia.geometry as kg
import numpy as np
import torch.nn.functional as F

class Projector():

    @staticmethod
    def pix2world(points, depth_map, poses, Ks):
        """Unprojects pixels to 3D coordinates."""
        H, W = depth_map.shape[2:]
        cam = Projector._make_camera(H, W, Ks, poses)
        depths = Projector._sample_depths(depth_map, points)
        return Projector._pix2world(points, depths, cam)

    @staticmethod
    def world2pix(points, res, poses, Ks, depth_map=None, eps=1e-2):
        """Projects 3D coordinates to screen."""
        assert len(Ks.shape) == 3
        cam = Projector._make_camera(res[0], res[1], Ks, poses)
        xy, depth = Projector._world2pix(points, cam)

        if depth_map is not None:
            depth_dst = Projector._sample_depths(depth_map, xy)
            xy[(depth < 0) | (depth > depth_dst + eps) | ((xy.abs() > 1).any(dim=-1))] = np.nan

        return xy, depth

    @staticmethod
    def _make_camera(height, width, K, pose):
        """Creates a PinholeCamera with specified intrinsics and extrinsics."""
        intrinsics = torch.eye(4, 4).to(K).repeat(len(K), 1, 1)
        intrinsics[:, 0:3, 0:3] = K

        extrinsics = torch.eye(4, 4).to(pose).repeat(len(pose), 1, 1)
        extrinsics[:, 0:4, 0:4] = pose

        height, width = torch.tensor([height]).to(K), torch.tensor([width]).to(K)

        return kg.PinholeCamera(intrinsics, extrinsics, height, width)

    @staticmethod
    def _pix2world(p, depth, cam):
        """Projects p to world coordinate.

        Args:
        p:     List of points in pixels (B, N, 2).
        depth: Depth of each point(B, N).
        cam:   Camera with batch size B

        Returns:
        World coordinate of p (B, N, 3).
        """
        p = distort_denormalize(p, cam.intrinsics[..., :3, :3])
        p_h = kg.convert_points_to_homogeneous(p)
        p_cam = kg.transform_points(cam.intrinsics_inverse(), p_h) * depth.unsqueeze(-1)
        return kg.transform_points(kg.inverse_transformation(cam.extrinsics), p_cam)

    @staticmethod
    def _world2pix(p_w, cam):
        """Projects p to normalized camera coordinate.

        Args:
        p_w: List of points in world coordinate (B, N, 3).
        cam: Camera with batch size B

        Returns:
        Normalized coordinates of p in pose cam_dst (B, N, 2) and screen depth (B, N).
        """
        proj = kg.compose_transformations(cam.intrinsics, cam.extrinsics)
        p_h = kg.transform_points(proj, p_w)
        p, d = kg.convert_points_from_homogeneous(p_h), p_h[..., 2]
        return undistort_normalize(p, cam.intrinsics[..., :3, :3]), d

    @staticmethod
    def _project_points(p, depth_src, cam_src, cam_dst):
        """Projects p visible in pose T_p to pose T_q.

        Args:
        p:                List of points in pixels (B, N, 2).
        depth:            Depth of each point(B, N).
        cam_src, cam_dst: Source and destination cameras with batch size B

        Returns:
        Normalized coordinates of p in pose cam_dst (B, N, 2).
        """
        return Projector._world2pix(Projector._pix2world(p, depth_src, cam_src), cam_dst)

    @staticmethod
    def _sample_depths(depths_map, points):
        """Samples the depth of each point in points"""
        assert depths_map.shape[:2] == (len(points), 1)
        return F.grid_sample(depths_map, points[:, None], align_corners=False)[:, 0, 0, ...]

def invert_Rt(Rt_mat):
    R, t = Rt_mat[..., 0:3, 0:3], Rt_mat[..., 0:3, 3:4]
    reps = tuple(Rt_mat.shape[:-2]) + (1, 1)
    if isinstance(Rt_mat, np.ndarray):
        out = np.tile(np.eye(4), reps)
        RT = np.swapaxes(R, -1, -2)
    elif isinstance(Rt_mat, torch.Tensor):
        out = torch.eye(4, device=Rt_mat.device).repeat(reps)
        RT = torch.swapaxes(R, -1, -2)
    else:
        raise ValueError(f"Unknown matrix type: {type(Rt_mat)}")

    out[..., 0:3, 0:3] = RT
    out[..., 0:3, 3:4] = -RT @ t
    return out

def distort_denormalize(
        coords: torch.Tensor, K: torch.Tensor, dist_coeff: Optional[torch.Tensor] = None, denormalize: bool = True):
    if dist_coeff is None:
        dist_coeff = torch.zeros(4)

    new_K = torch.eye(3).to(coords) if denormalize else None

    return kg.distort_points(coords, K.to(coords), dist_coeff.to(coords), new_K=new_K)


def undistort_normalize(
        coords: torch.Tensor, K: torch.Tensor, dist_coeff: Optional[torch.Tensor] = None, normalize: bool = True):
    if dist_coeff is None:
        dist_coeff = torch.zeros(4)
        num_iters = 0
    else:
        num_iters = 5

    new_K = torch.eye(3).to(coords) if normalize else None

    return kg.undistort_points(coords, K.to(coords), dist_coeff.to(coords), new_K=new_K, num_iters=num_iters)

def overlay_depth(keyframe, depth, pose_graph):
    depth_h, depth_w = depth.shape[2:]
    coord_src_normed = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / depth_h, 1 - 1 / depth_h, depth_h, device=device),
            torch.linspace(-1 + 1 / depth_w, 1 - 1 / depth_w, depth_w, device=device),
        ),
        indexing = 'ij',
    )
    pose_src = keyframe.pose
    K_src = keyframe.K
    points = torch.stack(coord_src_normed).flatten(1).mT.cpu()
    world = Projector.pix2world(points[None], depth, torch.tensor(invert_Rt(pose_src))[None].float(), torch.tensor(K_src)[None])
    world = world[0].cpu().numpy()
    for world_point, point in zip(world, points):
        keyframe.add_features(xy=point[None, [1, 0]].cpu().numpy(), info=[{'xy': (point[[1, 0]] * torch.tensor([depth_w, depth_h]))}])
        if np.isnan(world_point).any() or np.isinf(world_point).any():
            continue
        pose_graph.add_landmark(world_point, [keyframe.features[-1]])
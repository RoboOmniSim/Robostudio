# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.models.splatfacto_backward import SplatfactobackwardModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

import trimesh

from nerfstudio.robotic.utils.utils import load_transformation_package, relative_tf_to_global,load_txt,load_txt_bbox
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.physics_engine.python.collision_detection import collision_detection
from nerfstudio.robotic.kinematic.gripper_utils import *
from nerfstudio.robotic.physics_engine.omnisim.issac2sim import *
from nerfstudio.robotic.config.raw_config import Roboticconfig
@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""

@dataclass
class RoboExporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""
    experiment_type: str
    """Type of experiment to export. 'novelpose' 'push_bag' 'grasp' 'grasp_object' """
    output_file: Path
    """Path to the output file of trajectory from simulator."""

    static_path: Path
    """Path to the static object ply file."""



    use_gripper: bool=True
    """Whether to use the gripper for the export."""
    
    export_part: bool=False
    """Whether to export each part of the model to view segmentation quality."""
    
    time_stamp: int=1
    """Time stamp for the simulation."""


    load_bbox_info: Path=Path("./dataset/issac2sim/part/bbox_info/bbox_list.txt")
    """Path to the bounding box information file."""

    trajectory_file: Path=Path("./dataset/issac2sim/trajectory/dof_positions.txt")
    """Path to the trajectory file."""
    
    meta_sim_path: Path=Path("./dataset/issac2sim/meta_sim/meta_sim.yaml")
    """Path to the meta simulation file for omnisim adapter."""

def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        filename = self.output_dir / "splat.ply"

        map_to_tensors = {}

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            n = positions.shape[0]
            map_to_tensors["positions"] = positions
            map_to_tensors["normals"] = np.zeros_like(positions, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=1))
            n_after = np.sum(select)  

            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select, :]

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        o3d.t.io.write_point_cloud(str(filename), pcd)


@dataclass
class ExportGaussianSplat_mesh(RoboExporter):

    """
    Export 3D Gaussian Splatting model based on semantic mapping comes from bounding box to a .ply
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)


        # Define your bounding boxes as [xmin, ymin, zmin, xmax, ymax, zmax]


        # operation scene bbox

        operation_scene_bbox = np.array([[-1, -1, -1, 1, 1, 1],[-0.4,-0.3,-0.86,1.0,0.5,-0.04]])

        operation_scene_bbox_ids = np.array([0,1]) # 0 is background, 1 is the operation scene

        

        # export each part of the model to view segmentation quality 
        export_part = self.export_part

        # roboconfig way
        # update experiment with meta_sim_path
        roboconfig=Roboticconfig()
        roboconfig.setup_params(self.meta_sim_path)


        expand_bbox = roboconfig.expand_bbox
        contain_object = roboconfig.contain_object
        self.use_gripper = roboconfig.use_gripper

        experiment_type=roboconfig.experiment_type
        

        # raw
        experiment_type = self.experiment_type
        if experiment_type == "novelpose":
            expand_bbox=True
            self.use_gripper=False # no gripper for novelpose
            contain_object=False
        elif experiment_type == "push_box":
            expand_bbox=False
            self.use_gripper=False # no gripper for push_box
            contain_object=True
        elif experiment_type == "grasp":
            contain_object=False
            expand_bbox=False
            self.use_gripper=True # no gripper for push_box
        elif experiment_type == "grasp_object":
            contain_object=True
            expand_bbox=False
            self.use_gripper=True
        # the inside value is not important and it is just a placeholder during debug
        bboxes_gripper = np.array([
        [-1, -1, -1, 1, 1, 1],  # Bounding box 0 # all point is in bounded scene 
        [-0.304, 0.07, -0.652, -0.205, 0.19, -0.508],  # Bounding box 1
        [-0.302, -0.059, -0.64, -0.166, 0.082, -0.15],  # Bounding box 2
        [-0.265, 0.06, -0.3, 0.1, 0.17, -0.11],  # Bounding box 3
        [0, -0.02, -0.25, 0.127, 0.06, -0.12],   # Bounding box 4
        [0.051, -0.023, -0.18, 0.192, 0.047, -0.04],  # Bounding box 5
        [0.148, -0.018, -0.19, 0.206, 0.047, -0.13],  # Bounding box 6
        [0.184, -0.04, -0.27, 0.311, 0.07, -0.14],  # Bounding box 7
        [0, 0, 0, 0, 0, 0],  # Bounding box 8 object
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 9 base_link
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 10
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 11
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 12
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 13
        ])


        bboxes_nogripper = np.array([
        [-1, -1, -1, 1, 1, 1],  # Bounding box 0 # all point is in bounded scene 
        [-0.304, 0.07, -0.652, -0.205, 0.19, -0.508],  # Bounding box 1
        [-0.302, -0.059, -0.64, -0.166, 0.082, -0.15],  # Bounding box 2
        [-0.265, 0.06, -0.3, 0.1, 0.17, -0.11],  # Bounding box 3
        [0, -0.02, -0.25, 0.127, 0.06, -0.12],   # Bounding box 4
        [0.051, -0.023, -0.18, 0.192, 0.047, -0.04],  # Bounding box 5
        [0.148, -0.018, -0.19, 0.206, 0.047, -0.13],  # Bounding box 6
        [0.184, -0.04, -0.27, 0.311, 0.07, -0.14],  # Bounding box 7
        [0, 0, 0, 0, 0, 0],  # Bounding box 8
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 9
        ])

        # IDs for each bounding box
             
        bbox_ids_gripper = np.array([0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13])  # based on label map in the urdf file  0 is background, 1-6 are the linkage, 7 is the gripper center,  8 is the object,9 is base_link (don;t move actually), 10 is the gripper left down,11 is left up,12 is right down,13 is right up
        bbox_ids_nogripper = np.array([0,1, 2, 3, 4,5,6,7,8,9])  # based on label map in the urdf file  0 is background, 1-6 are the linkage, 7 is the gripper, 8 is the object,9 is base_link (don;t move actually)

        use_gripper=self.use_gripper
        if use_gripper:
            bboxes=bboxes_gripper
            bbox_ids=bbox_ids_gripper
        else:
            bboxes=bboxes_nogripper
            bbox_ids=bbox_ids_nogripper


        load_bbox_info=self.load_bbox_info
        bbox_list=np.loadtxt(load_bbox_info) 
        bbox_list=bbox_list.reshape(-1,6) # 12 total, 0 is base, 1-7 are link, 8-11 are rmatch with 10-13



        # rewrite this method to submethod 
        for i in range(len(bbox_list)):
            # replace the bboxes with the new scenes bboxs
            if use_gripper is False:
                if i==0:
                    bboxes[9]=bbox_list[i]
                elif i==1:
                    bboxes[1]=bbox_list[i]
                elif i==2:
                    bboxes[2]=bbox_list[i]
                elif i==3:
                    bboxes[3]=bbox_list[i]
                elif i==4:
                    bboxes[4]=bbox_list[i]
                elif i==5:  
                    bboxes[5]=bbox_list[i]
                elif i==6:
                    bboxes[6]=bbox_list[i]
                elif i==7:
                    bboxes[7]=bbox_list[i]


            else:

                if contain_object==False:# no object case
                    if i==0:
                        bboxes[9]=bbox_list[i]
                    elif i==1:
                        bboxes[1]=bbox_list[i]
                    elif i==2:
                        bboxes[2]=bbox_list[i]
                    elif i==3:
                        bboxes[3]=bbox_list[i]
                    elif i==4:
                        bboxes[4]=bbox_list[i]
                    elif i==5:  
                        bboxes[5]=bbox_list[i]
                    elif i==6:
                        bboxes[6]=bbox_list[i]
                    elif i==7:
                        bboxes[7]=bbox_list[i]
                    elif i==8:
                        bboxes[10]=bbox_list[i]
                    elif i==9:
                        bboxes[11]=bbox_list[i]
                    elif i==10:
                        bboxes[12]=bbox_list[i]
                    elif i==11:
                        bboxes[13]=bbox_list[i]

                # object case
                else:
                    if i==0:
                        bboxes[9]=bbox_list[i]
                    elif i==1:
                        bboxes[1]=bbox_list[i]
                    elif i==2:
                        bboxes[2]=bbox_list[i]
                    elif i==3:
                        bboxes[3]=bbox_list[i]
                    elif i==4:
                        bboxes[4]=bbox_list[i]
                    elif i==5:  
                        bboxes[5]=bbox_list[i]
                    elif i==6:
                        bboxes[6]=bbox_list[i]
                    elif i==7:
                        bboxes[7]=bbox_list[i]
                    elif i==8:
                        bboxes[8]=bbox_list[i]
                    elif i==9:
                        bboxes[10]=bbox_list[i]
                    elif i==10:
                        bboxes[11]=bbox_list[i]
                    elif i==11:
                        bboxes[12]=bbox_list[i]
                    elif i==12:
                        bboxes[13]=bbox_list[i]  

                



        # novelpose case
        if expand_bbox:
            # for manual bbox fix for the novelpose part when the manual bbox is not accurate


            # use a knn to make all point in region that belongs to background to its nerest point with sam id
            bboxes[3,1] = bboxes[3,1]+0.015   # expand the bounding box by 10% to ensure all points are included
            bboxes[3,3] = bboxes[3,3]+0.025 
            bboxes[3,0] = bboxes[3,0]*1.05
            bboxes[3,2] = bboxes[3,2]*1.05
            bboxes[3,4] = bboxes[3,4]*1.05
            bboxes[3,5] = bboxes[3,5]*1.05  
            bboxes[2,0] = bboxes[2,0]
            bboxes[2,3] = bboxes[2,3]+0.03
            bboxes[2,2] = bboxes[2,2]
            bboxes[2,5] = bboxes[2,5]+0.03

            bboxes[4,4] = bboxes[4,4]+0.015
            bboxes[4,3] = bboxes[4,3]+0.02
            
            bboxes[5,0] = bboxes[5,0]
            bboxes[5,3] = bboxes[5,3]+0.02
            bboxes[5,4] = bboxes[5,4]+0.035
            bboxes[:4,2] = bboxes[:4,2]*1.05

            bboxes[4:6,1]  = bboxes[4:6,1]*1.05




        




        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        dataparser_output=pipeline.datamanager.dataparser._generate_dataparser_outputs()


        global_transform = dataparser_output.dataparser_transform
        global_scale=dataparser_output.dataparser_scale








        filename = self.output_dir / "splat.ply"

        map_to_tensors = {}
        map_to_tensors_new = {}
        




       
        with torch.no_grad():
            positions = model.means.cpu().numpy()


            n = positions.shape[0]


            use_gcno = False
            use_sugar = False
            if use_gcno:
                normals=0 # the normals from the gcno
            elif use_sugar:
                normals=0   # the normals from the sugar
            else:
                normals=np.zeros_like(positions, dtype=np.float32)

                
            map_to_tensors["positions"] = positions
            map_to_tensors["normals"] = normals
            





            assigned_ids = np.zeros((n,4), dtype=int)


            for i, bbox in enumerate(bboxes):
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,0] = bbox_ids[i]


        
            group_id_4=4
            group_id_5=5
            group_id_6=6
            if group_id_4 == 4:
                i=group_id_4
                bbox=bboxes[group_id_4,:]
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,1] = bbox_ids[i]

            if group_id_5 == 5:
                i=group_id_5
                bbox=bboxes[group_id_5,:]
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,2] = bbox_ids[i]
            if group_id_6 == 6:
                i=group_id_6
                bbox=bboxes[group_id_6,:]
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,3] = bbox_ids[i]


            for i in range(assigned_ids.shape[0]):
                if assigned_ids[i,1] == 4 and assigned_ids[i,0] == 5:
                    assigned_ids[i,0] = assigned_ids[i,1]
                    
                if assigned_ids[i,3] == 6 and assigned_ids[i,0] == 5:
                    assigned_ids[i,0] = assigned_ids[i,1]



            assigned_ids_new = assigned_ids[:,0]
            map_to_tensors["semantic_id"] = np.array(assigned_ids_new.reshape(-1,1),dtype=np.float32)  # standard datastructure   





            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.int8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]
            
            
            
            

        # Select points with a specific ID (e.g., ID 1)



        if export_part:
            for index_id in range(len(bboxes)):
                group_id=index_id



                assigned_ids_new = assigned_ids[:,0]
                selected_points = positions[assigned_ids_new == group_id]
                map_to_tensors_new['positions'] = selected_points
                map_to_tensors_new['normals'] = normals[assigned_ids_new == group_id]
                map_to_tensors_new['semantic_id'] =  np.array(assigned_ids_new[assigned_ids_new == group_id].reshape(-1,1),dtype=np.float32)
                if model.config.sh_degree > 0:
                    shs_0 = model.shs_0.detach().contiguous().cpu().numpy()
                    for i in range(shs_0.shape[1]):
                        map_to_tensors_new[f"f_dc_{i}"] = shs_0[:, i, None][assigned_ids_new == group_id]

                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).detach().contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors_new[f"f_rest_{i}"] = shs_rest[:, i, None][assigned_ids_new == group_id]
                else:
                    colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                    map_to_tensors_new["colors"] = (colors * 255).astype(np.int8)[assigned_ids_new == group_id]

                map_to_tensors_new["opacity"] = model.opacities.data.cpu().numpy()[assigned_ids_new == group_id]

                scales = model.scales.data.cpu().numpy()
                for i in range(3):
                    map_to_tensors_new[f"scale_{i}"] = scales[:, i, None][assigned_ids_new == group_id]

                quats = model.quats.data.cpu().numpy()
                for i in range(4):
                    map_to_tensors_new[f"rot_{i}"] = quats[:, i, None][assigned_ids_new == group_id]


                pcd = o3d.t.geometry.PointCloud(map_to_tensors_new)
                
                part_filename=self.output_dir / f"splat_part_{group_id}.ply"
                o3d.t.io.write_point_cloud(str(part_filename), pcd)

        else:
            pcd = o3d.t.geometry.PointCloud(map_to_tensors)

            o3d.t.io.write_point_cloud(str(filename), pcd)



@dataclass
class ExportGaussianSplat_resampledmesh(RoboExporter):

    """
    Export 3D Gaussian Splatting model with gripper and full model id for our resampled mesh
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)



        # import from the urdf file

        # operation scene bbox

        operation_scene_bbox = np.array([[-1, -1, -1, 1, 1, 1],[-0.4,-0.3,-0.86,1.0,0.5,-0.04]])

        operation_scene_bbox_ids = np.array([0,1]) # 0 is background, 1 is the operation scene

        # Define your bounding boxes as [xmin, ymin, zmin, xmax, ymax, zmax]
        bboxes = np.array([
        [-1, -1, -1, 1, 1, 1],  # Bounding box 0 # all point is in bounded scene 
        [-0.304, 0.07, -0.652, -0.205, 0.19, -0.508],  # Bounding box 1
        [-0.302, -0.059, -0.64, -0.166, 0.082, -0.15],  # Bounding box 2
        [-0.265, 0.06, -0.3, 0.1, 0.17, -0.11],  # Bounding box 3
        [0, -0.02, -0.25, 0.127, 0.06, -0.12],   # Bounding box 4
        [0.051, -0.023, -0.18, 0.192, 0.047, -0.04],  # Bounding box 5
        [0.148, -0.018, -0.19, 0.206, 0.047, -0.13],  # Bounding box 6
        [0.184, -0.04, -0.27, 0.311, 0.07, -0.14],  # Bounding box 7
        [0.324, -0.03, -0.307, 0.352, 0.08, -0.25],  # Bounding box 8
        [-0.3, 0.08, -0.71, -0.2, 0.21, -0.63],  # Bounding box 9
        ])

        # IDs for each bounding box
        bbox_ids_gripper = np.array([0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13])  # based on label map in the urdf file  0 is background, 1-6 are the linkage, 7 is the gripper center,  8 is the object,9 is base_link (don;t move actually), 10 is the gripper left down,11 is left up,12 is right down,13 is right up
        bbox_ids = np.array([0,1, 2, 3, 4,5,6,7,8,9])  # based on label map in the urdf file  0 is background, 1-6 are the linkage, 7 is the gripper, 8 is the object,9 is base_link (don;t move actually)
        expand_bbox = True
        if expand_bbox:
            bboxes[:4,:] = bboxes[:4,:] * 1.05  # expand the bounding box by 10% to ensure all points are included







        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model


        


        dataparser_output=pipeline.datamanager.dataparser._generate_dataparser_outputs()


        global_transform = dataparser_output.dataparser_transforms
        global_scale=dataparser_output.dataparser_scale


        load_recenter_info_path ='/home/lou/gs/nerfstudio/Recenter_info.txt'
        recenter_info=load_txt(load_recenter_info_path)

        for i in range(len(recenter_info)):
            # replace the bboxes with the new scenes bboxs
            bboxes[i]=recenter_info[i]['gs_bbox_list']






        filename = self.output_dir / "splat.ply"

        map_to_tensors = {}
        map_to_tensors_new = {}
        


        if_deformable = False



        # pipeline should be remove all points that need to be deform


        # use is_deformed to deform, then add it back to the entire checkpoint to export



        # also if we want to retrain from a timestamp, just need to reload the deformed point in that scene


        # also in bbox but has no 0 id can be deleted (by the strict operation scene bbox, the one above is the coarse one)

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            colors=model.colors.cpu().numpy()



            n = positions.shape[0]


            dataparser = pipeline.datamanager.dataparser
            bbox_operation = np.array([[-1, -1, -1, 1, 1, 1]])
            get_resamplemesh=False
            if get_resamplemesh==True:
                mesh=model.get_mesh(dataparser,bbox_operation,map_to_tensors,colors)

            get_vdbmesh=True
            if get_vdbmesh==True:
                cameras=dataparser.get_cameras()
                mesh=model.get_vdbmesh(cameras)

            use_gcno = False
            use_sugar = False
            if use_gcno:
                normals=0 # the normals from the gcno
            elif use_sugar:
                normals=0   # the normals from the sugar
            else:
                normals=np.zeros_like(positions, dtype=np.float32)

                
            map_to_tensors["positions"] = positions
            map_to_tensors["normals"] = normals
            





            assigned_ids = np.zeros((n,4), dtype=int)


            for i, bbox in enumerate(bboxes):
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,0] = bbox_ids[i]


        
            group_id_4=4
            group_id_5=5
            group_id_6=6
            if group_id_4 == 4:
                i=group_id_4
                bbox=bboxes[group_id_4,:]
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,1] = bbox_ids[i]

            if group_id_5 == 5:
                i=group_id_5
                bbox=bboxes[group_id_5,:]
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,2] = bbox_ids[i]
            if group_id_6 == 6:
                i=group_id_6
                bbox=bboxes[group_id_6,:]
                in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
                assigned_ids[in_bbox,3] = bbox_ids[i]


            for i in range(assigned_ids.shape[0]):
                if assigned_ids[i,1] == 4 and assigned_ids[i,0] == 5:
                    assigned_ids[i,0] = assigned_ids[i,1]
                    
                if assigned_ids[i,3] == 6 and assigned_ids[i,0] == 5:
                    assigned_ids[i,0] = assigned_ids[i,1]


            # for i, bbox in enumerate(operation_scene_bbox):
            #     in_bbox = np.all((positions >= bbox[:3]) & (positions <= bbox[3:]), axis=1)
            #     assigned_ids[in_bbox] = operation_scene_bbox_ids[i]
            assigned_ids_new = assigned_ids[:,0]
            map_to_tensors["semantic_id"] = np.array(assigned_ids_new.reshape(-1,1),dtype=np.float32)  # standard datastructure   





            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.int8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]
            
            
            
            



        group_id=6



        assigned_ids_new = assigned_ids[:,0]
        selected_points = positions[assigned_ids_new == group_id]


        export_part = False
        if export_part:
            map_to_tensors_new['positions'] = selected_points
            map_to_tensors_new['normals'] = normals[assigned_ids_new == group_id]
            map_to_tensors_new['semantic_id'] =  np.array(assigned_ids_new[assigned_ids_new == group_id].reshape(-1,1),dtype=np.float32)
            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.detach().contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors_new[f"f_dc_{i}"] = shs_0[:, i, None][assigned_ids_new == group_id]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).detach().contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors_new[f"f_rest_{i}"] = shs_rest[:, i, None][assigned_ids_new == group_id]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors_new["colors"] = (colors * 255).astype(np.int8)[assigned_ids_new == group_id]

            map_to_tensors_new["opacity"] = model.opacities.data.cpu().numpy()[assigned_ids_new == group_id]

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors_new[f"scale_{i}"] = scales[:, i, None][assigned_ids_new == group_id]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors_new[f"rot_{i}"] = quats[:, i, None][assigned_ids_new == group_id]


            pcd = o3d.t.geometry.PointCloud(map_to_tensors_new)
            
            part_filename=self.output_dir / f"splat_part_{group_id}.ply"
            o3d.t.io.write_point_cloud(str(part_filename), pcd)

        else:
            pcd = o3d.t.geometry.PointCloud(map_to_tensors)

            o3d.t.io.write_point_cloud(str(filename), pcd)



@dataclass
class ExportGaussianSplat_mesh_deform(RoboExporter):

    """
    Export 3D Gaussian Splatting model based on sugar,gcno and part based semantic mapping to a .ply
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)



        # import from the urdf file

        # operation scene bbox

        operation_scene_bbox = np.array([[-1, -1, -1, 1, 1, 1],[-0.4,-0.3,-0.86,1.0,0.5,-0.04]])

        operation_scene_bbox_ids = np.array([0,1]) # 0 is background, 1 is the operation scene

        map_to_tensors = {}
        map_to_tensors_new = {}

        

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        dataparser_output=pipeline.datamanager.dataparser._generate_dataparser_outputs()


        global_transform = dataparser_output.dataparser_transform  # 3*4

        global_scale=dataparser_output.dataparser_scale

                    




        # omnisim way
        roboconfig=Roboticconfig()
        roboconfig.setup_params(self.meta_sim_path)


        experiment_type=roboconfig.experiment_type
        center_vector=roboconfig.center_vector
        scale_factor=roboconfig.scale_factor
        simulation_timestamp=roboconfig.simulation_timestamp
        add_simulation=roboconfig.add_simulation
        add_gripper=roboconfig.add_gripper
        start_time=roboconfig.start_time
        end_time_collision=roboconfig.end_time_collision
        flip_x_coordinate=roboconfig.flip_x_coordinate
        add_grasp_control=roboconfig.add_grasp_control
        add_grasp_object=roboconfig.add_grasp_object
        max_gripper_degree=roboconfig.max_gripper_degree
        add_trajectory=roboconfig.add_trajectory
        

        # raw
        experiment_type=self.experiment_type 
        output_file = self.output_file
        static_path=self.static_path

        if experiment_type=='novelpose':
            
            center_vector=np.array([-0.157,0.1715,-0.55]) #with base novel_pose

            scale_factor=np.array([1,1.25,1.65]) # x,y,z

            simulation_timestamp=0
            add_simulation=False
            add_gripper=False
            start_time=0
            end_time_collision=0.5
            flip_x_coordinate=False
            add_grasp_control=False
            add_grasp_object=False
            max_gripper_degree=-0.8525
        elif experiment_type=='push_bag':
            


            # center_vector=np.array([-0.261,0.145,-0.71]) #with base group1_bbox_fix push case

            # scale_factor=np.array([1.290,1.167,1.22]) # x,y,z
            
            center_vector=np.array([-0.261,0.138,-0.71]) #with base group1_bbox_fix push case

            scale_factor=np.array([1.290,1.167,1.22]) # x,y,z

            

            simulation_timestamp=1.12
            add_simulation=True
            add_gripper=False
            start_time=0
            end_time_collision=0.5
            flip_x_coordinate=False
            add_grasp_control=False
            add_grasp_object=False
            max_gripper_degree=-0.8525
        elif experiment_type=='issac2sim':
            


            center_vector=np.array([-0.261,0.138,-0.71]) #with base group1_bbox_fix push case

            scale_factor=np.array([1.290,1.167,1.22]) # x,y,z

            simulation_timestamp=1.12
            add_simulation=False
            add_gripper=True
            start_time=0
            end_time_collision=0.5
            flip_x_coordinate=False
            add_grasp_control=False
            add_grasp_object=True
            max_gripper_degree=-0.42
            add_trajectory=True


        elif experiment_type=='grasp':  # grasp data for the gripper only 
            center_vector=np.array([-0.135,0.1125,-0.78]) #with base grasp only case
            scale_factor=np.array([1.1,1.15,1.18]) # x,y,z
            add_simulation=False
            simulation_timestamp=0
            add_gripper=True
            start_time=0
            end_time_collision=1
            flip_x_coordinate=False
            add_grasp_control=True
            add_grasp_object=False
            max_gripper_degree=-0.8525
        elif experiment_type=='grasp_object':  # grasp data for the gripper and object


            center_vector=np.array([ 0.206349,    0.1249724, -0.70869258]) #with base grasp_object static fixed
            # center_vector_gt=np.array([  0.20764898,  0.15431145, -0.73875328]) #with base grasp_object dynamic
            scale_factor=np.array([1.2615,1.35,1.220]) # x,y,z

            add_simulation=False
            simulation_timestamp=0
            add_gripper=True
            start_time=0
            end_time_collision=0.5
            flip_x_coordinate=True
            add_grasp_control=True
            add_grasp_object=True
            max_gripper_degree=-0.8525
        else:
            print('experiment type not found')
            raise ValueError('experiment type not found')


        if add_gripper:
            
            # gripper_control,joint_angles_degrees_gripper, a_gripper, alpha_gripper, d_gripper=load_gripper_control(output_file,experiment_type,start_time=start_time,end_time=end_time_collision)
                    

            assigned_ids = np.array([0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13])  # with gripper
        else:
            assigned_ids = np.array([0,1, 2, 3, 4,5,6,7,8,9])  # no gripper
        # Read the pre timestamp angle state from txt file
        movement_angle_state,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt=load_uniform_kinematic(output_file,experiment_type,add_gripper=add_gripper,flip_x_coordinate=flip_x_coordinate,scale_factor_pass=scale_factor,center_vector_pass=center_vector)

        if_deformable = True

      

        if if_deformable:

            # the render should be all


            # total is 241 for push box


            
            time_stamp=self.time_stamp # the time stamp of the transformation package  # fps is 796/4=199 step from 3.33 sec in this bag with 60fps  random pick one deform here 
 
            scale_factor=dataparser_output.dataparser_scale  # replace by future 
            

            # if add_gripper:
            #     output_xyz, output_opacities, output_scales, output_features_extra, output_rots, output_features_dc,output_semantic_id=model.get_deformation(time_stamp,movement_angle_state,assigned_ids,
            #                                                                                                                                                  final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt,
            #                                                                                                                                                    add_gripper=add_gripper,path=static_path,add_simulation=add_simulation,dt=simulation_timestamp,flip_x_coordinate=flip_x_coordinate)
            
            if add_trajectory:

    
                
                traj_mode=edit_trajectory(self.trajectory_file,start_time)
                movement_angle_state=traj_mode

                

                # this is for issac2sim and the default is full control mode 
                add_grasp_control_value=max_gripper_degree
                add_grasp_object_control= 0 # this is for the different interaction time between the gripper and the object
                add_grasp_object_duration= (start_time,end_time_collision) # the object is follow gripper's move in this period
                output_xyz, output_opacities, output_scales, output_features_extra, output_rots, output_features_dc,output_semantic_id=model.get_deformation(time_stamp,movement_angle_state,assigned_ids,
                                                                                                                                                                final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt,
                                                                                                                                                                add_gripper=add_gripper,path=static_path,add_simulation=add_simulation,add_grasp_object=add_grasp_object,
                                                                                                                                                                add_grasp_object_duration=add_grasp_object_duration,
                                                                                                                                                                dt=simulation_timestamp,add_grasp_control=add_grasp_control_value,flip_x_coordinate=flip_x_coordinate,add_trajectory=add_trajectory)
                



            else:
                if add_gripper and add_grasp_control:
                    add_grasp_control_value=-0.4
                    add_grasp_object_control= 0 # this is for the different interaction time between the gripper and the object
                    add_grasp_object_duration= (start_time,end_time_collision) # the object is follow gripper's move in this period
                    output_xyz, output_opacities, output_scales, output_features_extra, output_rots, output_features_dc,output_semantic_id=model.get_deformation(time_stamp,movement_angle_state,assigned_ids,
                                                                                                                                                                final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,center_vector_gt,
                                                                                                                                                                add_gripper=add_gripper,path=static_path,add_simulation=add_simulation,add_grasp_object=add_grasp_object,
                                                                                                                                                                add_grasp_object_duration=add_grasp_object_duration,
                                                                                                                                                                dt=simulation_timestamp,add_grasp_control=add_grasp_control_value,flip_x_coordinate=flip_x_coordinate)
                
                
                else:
                    output_xyz, output_opacities, output_scales, output_features_extra, output_rots, output_features_dc,output_semantic_id=model.get_deformation(time_stamp,movement_angle_state,assigned_ids,final_transformations_list_0,scale_factor,a,alpha,d,joint_angles_degrees,
                                                                                                                                                                center_vector_gt,path=static_path,add_simulation=add_simulation,dt=simulation_timestamp)
                
            if add_simulation == True:
                filename = self.output_dir / f"splat_deform_timestamp{time_stamp}_simulation{simulation_timestamp}.ply"
            elif add_gripper:
                filename = self.output_dir / f"splat_deform_timestamp{time_stamp}_gripper.ply"
            else:
                
                filename = self.output_dir / f"splat_deform_timestamp{time_stamp}.ply"

        output_xyz=np.concatenate(output_xyz)
        output_opacities=np.concatenate(output_opacities)
        output_scales=np.concatenate(output_scales)
        output_features_extra=np.concatenate(output_features_extra)
        output_rots=np.concatenate(output_rots)
        output_features_dc=np.concatenate(output_features_dc)
        output_semantic_id=np.concatenate(output_semantic_id)
        # pipeline should be remove all points that need to be deform


        # use is_deformed to deform, then add it back to the entire checkpoint to export



        # also if we want to retrain from a timestamp, just need to reload the deformed point in that scene




        with torch.no_grad():



            use_gcno = False
            use_sugar = False
            if use_gcno:
                normals=0 # the normals from the gcno
            elif use_sugar:
                normals=0   # the normals from the sugar
            else:
                normals=np.zeros_like(output_xyz, dtype=np.float32)

                
            map_to_tensors["positions"] = output_xyz
            map_to_tensors["normals"] = normals
            


            map_to_tensors["semantic_id"] = np.array(output_semantic_id,dtype=np.float32)  # standard datastructure   






            
            for i in range(output_features_dc.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = output_features_dc[:, i]

                # transpose(1, 2) was needed to match the sh order in Inria version
            output_features_extra = output_features_extra.reshape((output_features_extra.shape[0], -1))
            for i in range(output_features_extra.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = output_features_extra[: , i, None]


            map_to_tensors["opacity"] = output_opacities

            

            # edit here 
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = output_scales[:, i, None]

            
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = output_rots[:, i, None]
            
            
            
            
            


                
       
        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        o3d.t.io.write_point_cloud(str(filename), pcd)

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportGaussianSplat_mesh, tyro.conf.subcommand(name="gaussian-splat-mesh")],
        Annotated[ExportGaussianSplat_mesh_deform, tyro.conf.subcommand(name="gaussian-splat-deformmesh")],
        Annotated[ExportGaussianSplat_resampledmesh, tyro.conf.subcommand(name="gaussian-splat-resampledmesh")],
        
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa

import torch.nn as nn
import open3d as o3d


from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F

import numpy as np
from os import makedirs, path
import os
from  PIL import Image

import math
import json
from nerfstudio.cameras.cameras import Cameras

# from nerfstudio.utils.io import load_from_json
def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)




def load_mask(mask_path):
    mask = o3d.io.read_image(mask_path)
    mask = torch.from_numpy(np.array(mask)).float()
    return mask





# read label map
def read_label_file(label_map_path):
    label_dict = {}
    with open(label_map_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(':')
                label = parts[0]
                color_rgb = np.array(tuple(map(int, parts[1].split(','))))
                label_dict[label] = {'color_rgb': color_rgb, 'label': label}
    return label_dict

file_path = '/home/lou/Downloads/robot6'





mask_file_path = os.path.join(file_path,'full_masks/SegmentationClass')

def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def load_images_to_dict(image_folder):
    image_dict = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add any other image formats you need
            image_path = os.path.join(image_folder, filename)
            pil_image = Image.open(image_path)

            image_array = np.array(pil_image, dtype="uint8") 

            name=filename[:11]
            image_dict[name] = image_array
    return image_dict






def process_camera(meta):
    poses = []

        
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    fx.append(meta["fl_x"])
    fy.append(meta["fl_y"])
    cx.append(meta["cx"])
    cy.append(meta["cy"])
    height.append(1076)
    width.append(1914)
    fnames = []
    for frame in meta["frames"]:
        filepath = frame["file_path"]
        data_dir='images'
        fname = filepath[7:]
        fnames.append(fname)
    inds = np.argsort(fnames)
    frames = [meta["frames"][ind] for ind in inds]

    for frame in frames:
        pose = np.array(frame["transform_matrix"])
        poses.append(pose)


    return fx[0], fy[0], cx[0], cy[0], height[0], width[0], poses,fnames,inds







# Replace 'your_image_folder_path' with the path to your folder containing images

images_dict = load_images_to_dict(mask_file_path)



label_map_path=os.path.join(file_path,'full_masks/labelmap.txt')
label_dict = read_label_file(label_map_path)

# Accessing the label information
for label, info in label_dict.items():
    print(f"Label: {label}, Color RGB: {info['color_rgb']}, Parts: {info['label']}")







source_path='/home/lou/Downloads/robot6/robot6/transforms.json'

meta = load_from_json(source_path)


#inference the gs for the entire cameras




fx, fy, cx, cy, height, width, poses,fnames,inds=process_camera(meta)



output_path='/home/lou/Downloads/robot6/'
load_gt_images=False
# iterate all cam_idx
    
#cam_idx is colmap_id here
max_sh_degree=3



def load_ply(path,max_sh_degree):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
    # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # features_dc = np.zeros((xyz.shape[0], 3, 1))
    # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    # for idx, attr_name in enumerate(extra_f_names):
    #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    # scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    # scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    # scales = np.zeros((xyz.shape[0], len(scale_names)))
    # for idx, attr_name in enumerate(scale_names):
    #     scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    # rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    # rots = np.zeros((xyz.shape[0], len(rot_names)))
    # for idx, attr_name in enumerate(rot_names):
    #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    # features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    # scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    # rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    # return xyz, features_dc, features_rest, opacity, scaling, rotation
    return xyz

mesh_path=os.path.join(file_path,'result/splat.ply')



# this is only for point cloud we also need a mesh version
xyz=load_ply(mesh_path,max_sh_degree)



points=xyz # pytorch3d to opencv
# add a training_cameras here 





# build up the colmap_id   image_id mapping

# rename mask by the colmap_id image_id pair to the p3d_cameras ordering





def ndc2pix(x, W, c):
    return 0.5 * W * x  + c - 0.5


def project_pix(fullmat, p, img_size, center, eps=1e-6):
    p_hom = F.pad(p, (0, 1), value=1.0)
    p_hom = torch.einsum("...ij,...j->...i", fullmat, p_hom)
    rw = 1.0 / (p_hom[..., 3] + eps)
    p_proj = p_hom[..., :3] * rw[..., None]
    u = ndc2pix(p_proj[..., 0], img_size[0], center[0])
    v = ndc2pix(p_proj[..., 1], img_size[1], center[1])
    return torch.stack([u, v], dim=-1)












# nerfstudio only

def projection_matrix(fx, fy, W, H, n=0.01, f=1000.0):
    return torch.tensor(
        [
            [2.0 * fx / W, 0.0, 0.0, 0.0],
            [0.0, 2.0 * fy / H, 0.0, 0.0],
            [0.0, 0.0, (f + n) / (f - n), -2 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

sam_list=np.zeros_like(points[:,0])

pix_list = [[]]*len(fnames)









pixel_mask_color= [[]]*len(fnames)
for cam_idx in range(len(fnames)):
    # cam_idx=100
    image_name=fnames[cam_idx][:11]
    #camera_idx is the colmap_id so fix it 
    
    mask=images_dict[image_name]

    
        
    pose_inx=inds[cam_idx]

    # scale_factor = 1.0
    # scale_factor /= float(np.max(np.abs(poses[pose_inx][:3, 3])))
    # poses [pose_inx][:3, 3] *= scale_factor

    R = poses[pose_inx][:3, :3]  # 3 x 3
    T = poses[pose_inx][:3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R_edit = np.diag(np.array([1, -1, -1]))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = np.eye(4)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    # calculate the FOV of the camera given fx and fy, width and height
    cx = cx
    cy = cy
    fovx = 2 * math.atan(width / (2 * fx))
    fovy = 2 * math.atan(height / (2 * fy))

    image_size = (width,height)
    # projmat = projection_matrix(0.001, 1000, fovx, fovy)

    projection_matrix_raw=projection_matrix(fx, fy, width, height)


    # # proj_mat = viewmat

    
    # # or 
    proj_mat=projection_matrix_raw @ viewmat

    index_full_list=[[]]*points.shape[0]
    color_full_list=[[]]*points.shape[0]
    # proj_mat=full_proj_transform
    for i in range(points.shape[0]):
        #project the points to the image plane
        single_means3d=points[i,:]  # 3d point of guassian center 
        # 
        proj_mat = proj_mat.clone().detach().to(
            device="cuda", dtype=torch.float32
        )
        single_means3d = torch.tensor(single_means3d, dtype=torch.float32, device="cuda")
        image_size = image_size
        
        center = torch.tensor([cx, cy], dtype=torch.float32, device="cuda")

        pix = project_pix(proj_mat, single_means3d, image_size,center)
        
        pix = pix.round().long()
        index_0=pix[0].clone().detach().cpu().numpy()
        index_1=pix[1].clone().detach().cpu().numpy()
        if index_0 < 0 or index_0 >= mask.shape[0] or index_1 < 0 or index_1 >= mask.shape[1]:
            continue
        else:

            print(
                f"Point {i} projected to pixel [{index_0}, {index_1}]"
            )
        #This part need improve




        # sam_id=label_dict(pixel_mask_color)
            index_full=np.array((index_0,index_1))
            index_full_list[i]=index_full
            color_full_list[i]=mask[index_0, index_1]

            # if mask[index_1, index_0] != [234.32,74]:
            #     print('mask color is not in bg')
    
    
    pix_list.append(index_full_list)
    pixel_mask_color.append(color_full_list) # figure out u,v and H, W relation


    

            # sam_list[i].append(sam_id)







def process_sam_points(gs_points_withcolor,label_dict):

    #leave only one mask of this point
    num_classes=len(label_dict)
    one_hot_labels = gs_points_withcolor

    # Sum the one-hot encoded array along the rows to get the count for each class
    class_counts = np.sum(one_hot_labels, axis=0)

    # Find the index of the most common mask
    most_common_mask = np.argmax(class_counts)

    return most_common_mask




most_common_mask=process_sam_points(sam_list,label_dict)










# now we can only export the point with the sam mask id we want



def export_part_ply(self, output_path, part_id):
    # Export the part to a ply file
    part_points = self.points[self.points.mask_id == part_id]
    part_colors = self.colors[self.points.mask_id == part_id]
    write_ply(output_path, part_points, part_colors, self.normals[self.points.mask_id == part_id])


def export_mask_whole_ply(self, output_path, mask_id):

    # for viewing only, we need to use the maskid to show the expect color of each gs 

    # Export the mask to a ply file
    mask_points = self.points[self.points.mask_id == mask_id]
    mask_colors = self.colors[self.points.mask_id == mask_id]
    write_ply(output_path, mask_points, mask_colors, self.normals[self.points.mask_id == mask_id])




def save_ply(self, path):
    mkdir_p(os.path.dirname(path))

    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
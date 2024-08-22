import os

import imageio
import numpy as np
import torch
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


# downsample gt video to 10 fps





def get_scene_images_tracking(l_image_filename):
    imgs = imageio.imread(l_image_filename)
    imgs = torch.from_numpy(
        (np.maximum(np.minimum(np.array(imgs), 255), 0) / 255.0).astype(np.float32)
    )
    return imgs[:, :, :3]


def cal_lpips(gt_path, zip_path, scale, scene_name):
    with torch.no_grad():
        lpips = LearnedPerceptualImagePatchSimilarity(
            normalize=True, net_type="vgg"
        ).cuda()
        device = torch.device("cuda:0")
        gt_suffix = f"_d{scale}.png"
        lpips_sum = 0
        for image in os.listdir(zip_path):
            if image.endswith(".png"):
                gt = torch.moveaxis(
                    get_scene_images_tracking(
                        os.path.join(gt_path, f"{image[-7:-4]}{gt_suffix}")
                    ),
                    -1,
                    0,
                )[None, ...].to(device)
                zip = torch.moveaxis(
                    get_scene_images_tracking(os.path.join(zip_path, image)), -1, 0
                )[None, ...].to(device)
                print(
                    f"calculating lpips for, gt_path:{os.path.join(gt_path, f'{image[-7:-4]}{gt_suffix}')}, zip_path:{os.path.join(zip_path, image)}"
                )
                lpips_sum += lpips(zip, gt)
                del zip
                del gt
                torch.cuda.empty_cache()
        lpips = lpips_sum / 200
        with open(f"cuda_lpips.txt", "a") as file:
            file.write(f"scene: {scene_name}, scale: {scale}, lpips: {lpips}")


for scene in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
    cal_lpips(GT_PATH_PREFIX + scene, ZIP_PATH_PREFIX + scene + "/test_preds", 0, scene)






import os

import imageio
import numpy as np
import torch
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio


def get_scene_images_tracking(l_image_filename):
    imgs = imageio.imread(l_image_filename)
    imgs = torch.from_numpy(
        (np.maximum(np.minimum(np.array(imgs), 255), 0) / 255.0).astype(np.float32)
    )
    return imgs[:, :, :3]


gt_prefix = "gt_d"
rip_prefix = "rip_d"
tri_prefix = "tri_d"
zip_prefix = "zip_d"

CROP = True


print(path)

cal_psnr = PeakSignalNoiseRatio(data_range=1.0)

gt_d0 = torch.moveaxis(get_scene_images_tracking(path + gt_prefix + "0.png"), -1, 0)[
    None, ...
]
rip_d0 = torch.moveaxis(get_scene_images_tracking(path + rip_prefix + "0.png"), -1, 0)[
    None, ...
]
tri_d0 = torch.moveaxis(get_scene_images_tracking(path + tri_prefix + "0.png"), -1, 0)[
    None, ...
]
zip_d0 = torch.moveaxis(get_scene_images_tracking(path + zip_prefix + "0.png"), -1, 0)[
    None, ...
]

gt_d1 = torch.moveaxis(get_scene_images_tracking(path + gt_prefix + "1.png"), -1, 0)[
    None, ...
]
rip_d1 = torch.moveaxis(get_scene_images_tracking(path + rip_prefix + "1.png"), -1, 0)[
    None, ...
]
tri_d1 = torch.moveaxis(get_scene_images_tracking(path + tri_prefix + "1.png"), -1, 0)[
    None, ...
]
zip_d1 = torch.moveaxis(get_scene_images_tracking(path + zip_prefix + "1.png"), -1, 0)[
    None, ...
]

gt_d2 = torch.moveaxis(get_scene_images_tracking(path + gt_prefix + "2.png"), -1, 0)[
    None, ...
]
rip_d2 = torch.moveaxis(get_scene_images_tracking(path + rip_prefix + "2.png"), -1, 0)[
    None, ...
]
tri_d2 = torch.moveaxis(get_scene_images_tracking(path + tri_prefix + "2.png"), -1, 0)[
    None, ...
]
zip_d2 = torch.moveaxis(get_scene_images_tracking(path + zip_prefix + "2.png"), -1, 0)[
    None, ...
]

gt_d3 = torch.moveaxis(get_scene_images_tracking(path + gt_prefix + "3.png"), -1, 0)[
    None, ...
]
rip_d3 = torch.moveaxis(get_scene_images_tracking(path + rip_prefix + "3.png"), -1, 0)[
    None, ...
]
tri_d3 = torch.moveaxis(get_scene_images_tracking(path + tri_prefix + "3.png"), -1, 0)[
    None, ...
]
zip_d3 = torch.moveaxis(get_scene_images_tracking(path + zip_prefix + "3.png"), -1, 0)[
    None, ...
]

print(f"d0 rip psnr {cal_psnr(gt_d0, rip_d0)}")
print(f"d0 tri psnr {cal_psnr(gt_d0, tri_d0)}")
print(f"d0 zip psnr {cal_psnr(gt_d0, zip_d0)}")

print(f"d1 rip psnr {cal_psnr(gt_d1, rip_d1)}")
print(f"d1 tri psnr {cal_psnr(gt_d1, tri_d1)}")
print(f"d1 zip psnr {cal_psnr(gt_d1, zip_d1)}")

print(f"d2 rip psnr {cal_psnr(gt_d2, rip_d2)}")
print(f"d2 tri psnr {cal_psnr(gt_d2, tri_d2)}")
print(f"d2 zip psnr {cal_psnr(gt_d2, zip_d2)}")

print(f"d3 rip psnr {cal_psnr(gt_d3, rip_d3)}")
print(f"d3 tri psnr {cal_psnr(gt_d3, tri_d3)}")
print(f"d3 zip psnr {cal_psnr(gt_d3, zip_d3)}")



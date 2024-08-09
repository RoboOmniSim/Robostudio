from nerfstudio.process_data.colmap_utils import colmap_to_json


import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import appdirs
import cv2
import numpy as np
import torch
from rich.progress import track

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
)
from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

# this scripte aims to transform colmap result to json nerfstudio format

if __name__ == "__main__":
    recon_path = Path("/home/lou/Downloads/sparse(1)/sparse/0")
    output_dir = Path("/home/lou/Downloads/grasp_object_colmap_modintrinsic")
    number_of_frame=colmap_to_json(recon_dir=recon_path, output_dir=output_dir)
    print(f"Number of frames: {number_of_frame}")
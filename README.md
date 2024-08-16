<div align="center"><h2>RoboStudio: A Physics Consistent World Model for Robotic Arm with Hybrid Representation</h2></div>


<p align="center">
    <a href='https://open-air-sun.github.io/mars/static/data/MARS-poster.pdf'>
        <img src='https://img.shields.io/badge/docs-passing-pink'/>
    <!-- community badges -->
    <a href="https://open-air-sun.github.io/mars/"><img src="https://img.shields.io/badge/Project-Page-ffa"/></a>
    <!-- doc badges -->
    <a href="https://arxiv.org/abs/2307.15058">
        <img src='https://img.shields.io/badge/arXiv-passing-aff'>
    </a>
    <a href="https://pypi.org/project/mars-nerfstudio/"><img src="https://img.shields.io/badge/pypi package-0.3.4-brightgreen"></a>

</p>


<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/logo-rs.png" />
    <img alt="tyro logo" src="docs/_static/imgs/logo-rs.png" width="30%"/>
</picture>
</div>


<div align="center">
  <img alt="" src="https://github.com/RoboOmniSim/Robostudio/assets/107318439/0d1cbe74-a2a3-4a94-af78-315b8a44ccb1">
</div>





# About

_It’s as simple as plug and play with robostudio!_


# Quickstart

<!-- TODO: replace the gaussian training and render process with omnisim config stucture -->

<!-- TODO(Yiran): Maybe code from nerfstudio should remove, user can download themselves? -->

## 1. Installation: Setup the environment

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name robostudio -y python=3.8
conda activate robostudio
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118


conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing Robostudio

Easy option:

```bash
pip install robostudio
```

**OR** if you want the latest and greatest:

```bash
git clone https://github.com/RoboOmniSim/Robostudio.git
cd Robostudio
pip install setuptools==69.5.1
pip install -e .
```

#### If you want to use our full robotic engine, please install pypose 
```bash
pip install pypose
```

## 2.Build up Dynamic physical consistent Gaussian Splatting asset 


static dataset (Gaussian splatting) 

find the dataset you want to follow:(grasp,zero_pose,novelpose,grasp_object,push_bag)

--data is path of dataset
```bash
ns-train  splatfacto --data 
```




find base and scale manually based on exported ply file of static gaussian splatting

```bash
ns-export  gaussian-splat --load-config --output-dir
```


get bounding box by hand 
```bash
python nerfstudio/robotic/export_util/load_bbox_from_part.py --full_bbox_path  --save_path
```

get bounding box list based on base and scale
```bash
python nerfstudio/robotic/export_util/export_bbox_withgripper.py --load_path 
```

### we use push box case as an exmaple (For full command, you can refer to launch.json)

export part gaussian splatting and semantic ply

--load-config is the path to the static output

--output-dir is the path to save the semantic model

--experiment_type specific the experiment type

--output_file is the trajectory file from moveit or omnisim

--static_path is the path to the origincal static gaussian ply output

--load_bbox_info is the path to bounding box information by hand or scripts

--export_part is the boolean varible to export part gaussian ply or export full semantic ply

--use_gripper is the boolean varible to use gripper or not 

```bash
ns-export gaussian-splat-mesh
        --load-config=
        --output-dir=
        --experiment_type=push_bag
        --output_file=
        --static_path=
        --load_bbox_info=./dataset/issac2sim/part/bbox_info/bbox_list.txt
        --export_part=False
        --use_gripper=True
        --meta_sim_path=./config_info/push_bag.yaml
```



### load trajectory from omnisim or real world application


Omnisim or Moveit 

We provide the exported result in the dataset, if you want to export based on your moveit scenes, 
you can follow the scripts in Robostudio/nerfstudio/robotic/ros


### forward rendering and simulation

export deform for single timestamp




 --time_stamp the time_stamp of output scenes 


```bash
ns-export gaussian-splat-deformmesh

        --load-config=
        --output-dir=exports/splat/no_downscale/group1_bbox_fix/correct_kinematic
        --experiment_type=push_bag
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt
        --time_stamp=
        --meta_sim_path=./config_info/push_bag.yaml
```


render for novel-trajectory and novel-time

```bash
ns-render  dynamic_dataset
        --load-config=
        --output_path=renders/push_box_dynamic
        --experiment_type=push_bag
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt
        --meta_sim_path=./config_info/push_bag.yaml

```







## 3. Advanced Options



### Obtain URDF from video


#### Convert dataset from nerfstudio to 2dgs

```bash
python nerfstudio/robotic/export_util/2dgs_utils/nerfstudioconvert2dgs.py -s "your data" --skip_matching
```

#### Train 2dgs and obtain mesh
You can follow the instruction from https://github.com/hugoycj/2.5d-gaussian-splatting
We will merge this part to our wheels after the PR of nerfstudio with 2dgs merged 

#### Fix orientation
Since the result of colmap is not necessary axis-aligned
You can use this command to fix it 

```bash
python nerfstudio/robotic/export_util/reorient.py 
--input_mesh_path
--output_mesh_path
--re_orientation_matrix
```

re_orientation_matrix is the transform matrix from nerfstudio dataparser to rescale and reorient the mesh


#### Get part 

First method: You can obtain part by base and scale 
Second method: manual bounding box(recommend manual bounding box for complex scenes and high accuracy)
Third method: Use SAM reproject Gaussian or SegAnyGAussians( https://github.com/Jumpat/SegAnyGAussians)

#### Load part to URDF



We first need to remap part to origin in our uniform coordinate

```bash
python nerfstudio/robotic/export_util/urdf_utils/urdf.py --part_path ./dataset/roboarm2/urdf/2dgs/arm --save_path ./dataset/roboarm2/urdf/2dgs/recenter_mesh --kinematic_info_path ./config_info/kinematic_info.yaml --experiment_type cr3 --scale_factor_gt 1.0 --num_links 8 --original_path dataset/roboarm2/roboarm2/urdf/2dgs/original_link

```

You can pick the prefer optimization method in kinematic_info.yaml config file 

##### Tricks for optimize URDF
Since there are Gap between design and real world robot, due to the error in installation and reconstruction, you may need to manual edit some part of DH parameter for best result, we estimate this takes you 10 mins.
The step should be optimize a and d based on real world scale, then fix alpha based on different rotation axis representation
Last step is to fix the base axis of reconstructed scenes and the part orientation

#### load the part mesh to urdf
load recenter mesh, scale, axis, initial pose and object mesh information to Omnisim

You can load the part mesh path to urdf with both collision mesh and visual mesh
Our model contain pre-point color for visual mesh, and you can decide to whether use it or no

#### setup object and robotic arm hyper parameter from LLM

The hyper parameter of object and robotic arm can be obtained by LLM 
You can email me for exact prompt










## use omnisim-issac backend to implement policy 

Use pre-build or custom urdf and fix collision group based on omnisim
```bash
python sim\metasim\real2isaac\real2isaac_grasp_v4.py
```
This is example of our pre-designed grasping policy 

### video2policy2real
export policy to real world in format of gripper pose





### video2policy2render
export policy to Gaussian Scenes in format of trajectory


### render result 
<div align="center">
  <img alt="" src="https://github.com/RoboOmniSim/Robostudio/assets/42707859/87804e24-693d-42dd-8b04-e537d7ed329a">
</div>




 --trajectory_file the trajectory from issac2sim only

This export the issac trajectory for certain time-stamp
```bash
ns-export gaussian-splat-deformmesh

        --load-config=
        --output-dir=exports/splat/no_downscale/group1_bbox_fix/correct_kinematic
        --experiment_type=issac2sim
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt
        --time_stamp=
        --meta_sim_path=./config_info/push_bag.yaml
```


render for novel-trajectory and novel-time

```bash
ns-render  dynamic_dataset
        --load-config=
        --output_path=renders/push_box_dynamic
        --experiment_type=issac2sim
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt
        --meta_sim_path=./config_info/push_bag.yaml

```

## 4. Dataset 


You can find data in this drive link: https://drive.google.com/file/d/1aJLMQjY0BOL-Wny6bXzNHq8unOr3mmjL/view?usp=drive_link

For dataset of URDF production, you can email me haozhelo@usc.edu

## Function of Gaussian Based robotic arm

### Grasping
<div align="center">
  <img alt="" src="https://github.com/RoboOmniSim/Robostudio/assets/42707859/ae4b3bfb-6f7e-4cd4-a647-9075ee8c051d">
</div>

### novel pose 
<div align="center">
  <img alt="" src="https://github.com/RoboOmniSim/Robostudio/assets/42707859/c5354df5-774d-4bdd-a5e5-7f3776fc1740">
</div>

### Backward optimized result
<div align="center">
  <img alt="" src="https://github.com/RoboOmniSim/Robostudio/assets/42707859/25d96e79-6ae0-4f99-875f-20fae500444b">
</div>



## 5: Explanation of Config file and how to connect it with omnisim

The config file are designed for each experiment
For the meaning of command, you can refer to the documentation of Roboconfig

You can follow our instruction and edit the config file for new Dataset 


YOU can use export_urdf_to_omnisim_config to generate urdf and simulation from Robostudio to Omnisim

You can use omni2gs_config to load the policy result from Omnisim to Gaussian Splatting




# Built On
<a href="https://github.com/nerfstudio-project/nerfstudio">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/docs/_static/imgs/logo-dark-ns.png" />
    <img alt="tyro logo" src="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/docs/_static/imgs/logo-ns.png" width="150px" />
</picture>
</a>

 
# Citation

TODO
You can find a paper writeup of the framework on [arXiv](https://arxiv.org/abs/2302.04264).

If you use this library or find the documentation useful for your research, please consider citing:

TODO
```
@inproceedings{nerfstudio,
	title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
	author       = {
		Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
		and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
		Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
		Angjoo
	},
	year         = 2023,
	booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
	series       = {SIGGRAPH '23}
}
```

# Contributors

<a href="https://github.com/RoboOmniSim/Robostudio/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=RoboOmniSim/Robostudio" />
</a>


# Acknowledgement
We want to thanks for the great help from Zitong Zhan, Ruilong Li, Junchen Liu, Zirui Wu, Yuantao Chen, Zhide Zhong, and Baijun Ye.
This pipeline inspired from the talk of Professor Hao Su, Professor Hongjing Lu and Professor Yaqin Zhang.


# Future Work
## Full Backward engine
We design a backward engine respect to kinematic and dynamic and we test it on the optimization of mass of object. But we find that our experiment is unfair and the kinematic of robotic arm is not good enough by backward. We will keep working on this and release it in future.

## End-to-end semantic tools
Release before 2025.1.1 due to potential progress in Gaussian based semantic labeling

## Motion-retargeting
We find the the editing functionality of our work can be used for motion retargeting 
We will release this part by 2024.12.01

## 4D interactable viewer 
Current 4D Gaussian Viewer cannot perform our full functionality, so we are designing a custom viewer and make it compatible with OmniSim.

## 6 DOF tracking
In traditional Reinforcement Learning pipeline, we use pixel level object tracking.
However, recent work like foundationpose reveal the possibility of video based object 6 DOF tracking
Our Gaussian-Pixel-Mesh binding can help to build up consistency between 6 DOF tracking. policy and rendering.

## More supported Simulation like articulated object, soft body and etc...



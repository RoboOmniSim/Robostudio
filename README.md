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

<!-- replace the nerfstudio logo with Robostudio logo -->

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

TODO: what will the below content go through

TODO(Yiran): Maybe code from nerfstudio should remove, user can download themselves?

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


## 2.Build up Dynamic physical consistent Gaussian Splatting asset 


static dataset (Gaussian splatting) 

find the dataset you want to follow:(grasp,zero_pose,novelpose,grasp_object,push_bag)

```bash
ns-train  splatfacto --data --vis+wandb
```




find base and scale manually by follow the exported ply file of static gaussian splatting

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

# we use push box case as an exmaple

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
```



# load trajectory from omnisim or real world application


omnisim or moveit 

We provide the exported result in the dataset, if you want to export based on your moveit scenes, 
you can follow the scripts in Robostudio/nerfstudio/robotic/ros


# forward rendering and simulation

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
```


render for novel-trajectory and novel-time

```bash
ns-export  dynamic_dataset
        --load-config=
        --output_path=renders/push_box_dynamic
        --experiment_type=push_bag
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt

```



<!-- Backward optimization to refine trajectory and physics parameter -->

<!-- load refined parameter to the omnisim -->





## 3. Advanced Options

obtain urdf from video

### we use zero-pose data to export urdf
2dgs obtain robotic arm mesh 

convert dataset from nerfstudio to 2dgs

```bash
python nerfstudio/robotic/export_util/2dgs_utils/nerfstudioconvert2dgs.py -s "your data" --skip_matching
```

Train 2dgs and obtain mesh


get part based on either base and scale or manual bounding box(recommend manual bounding box)



infer parameter from LLM 








## use omnisim-issac backend to implement policy 

export to urdf and fix collision group based on omnisim

video2policy2real
export policy to real world in format of gripper pose





video2policy2render
export policy to Gaussian Scenes in format of trajectory



 --trajectory_file the trajectory from issac only

```bash
ns-export gaussian-splat-deformmesh

        --load-config=
        
        --output-dir=exports/splat/no_downscale/group1_bbox_fix/correct_kinematic
        --experiment_type=issac2sim
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt
        --time_stamp=
```


render for novel-trajectory and novel-time

```bash
ns-export  dynamic_dataset
        --load-config=
        --output_path=renders/push_box_dynamic
        --experiment_type=issac2sim
        --output_file=
        --static_path=
        --trajectory_file=./dataset/issac2sim/trajectory/dof_positions.txt

```

## 4. Dataset 


We will upload four set of data 


and will add more data shortly



### Tensorboard / WandB / Viewer

We support four different methods to track training progress, using the viewer[tensorboard](https://www.tensorflow.org/tensorboard), [Weights and Biases](https://wandb.ai/site), and ,[Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=github). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command. Simultaneously utilizing the viewer alongside wandb or tensorboard may cause stuttering issues during evaluation steps. The viewer only works for methods that are fast (ie. nerfacto, instant-ngp), for slower methods like NeRF, use the other loggers.

# Learn More

And that's it for getting started with the basics of Robostudio.




# Supported Features


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

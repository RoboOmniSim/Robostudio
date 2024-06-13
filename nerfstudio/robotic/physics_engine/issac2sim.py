import os
import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
from nerfstudio.robotic.kinematic.uniform_kinematic import *
import torch
import argparse



def extract_values(string_list):
    values = []
    for s in string_list:
        # Remove unwanted characters
        clean_s = s.replace('[', '').replace(']', '').strip()
        # Convert to float and add to the values list
        values.append(float(clean_s))
    return values

def load_trajectory(path):
    with open(path, 'r') as file:
        data = file.read()
    
    # Split the data by 'Dof Positions:'
    blocks = data.split('Dof Positions:')[1:]
    
    # Parse each block of data
    dof_positions = []
    for block in blocks:
        # Extract the values from the block
        values = block.strip().strip('[]').split('], [')
        positions = [list(extract_values(value.split(','))) for value in values]
        dof_positions.append(positions)
    
    return dof_positions

def linear_interpolation(start, end, steps):
    return np.array([start + (end - start) * i / steps for i in range(steps)])


if __name__ == "__main__":
    filename = './dataset/issac2sim/trajectory/dof_positions.txt'

    # Read and print the DoF positions
    dof_positions = np.array(load_trajectory(filename)).flatten().reshape(-1, 6)
    for idx, positions in enumerate(dof_positions):
        print(f"Dof Positions {idx+1}:")
        for pos in positions:
            print(pos)
        print()
import trimesh
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score

def load_and_process_mesh(file_path, target_vertex_count=30000, scale_factor=400):
    # Load the mesh
    mesh = trimesh.load(file_path) # stl, ply and obj 
    
    # Downsample the mesh
    if len(mesh.vertices) > target_vertex_count:
        mesh = mesh.simplify_quadratic_decimation(target_vertex_count)
    
    # Recenter the mesh to the origin
    mesh.vertices -= mesh.centroid
    
    # Scale the mesh
    mesh.vertices *= scale_factor
    
    return mesh

def compute_mse_and_fscore(mesh1, mesh2):
    # Flatten the vertex arrays
    vertices1 = mesh1.vertices.flatten()
    vertices2 = mesh2.vertices.flatten()
    
    # Compute MSE
    mse = mean_squared_error(vertices1, vertices2)
    
    # Compute F-score
    f_score = f1_score(vertices1, vertices2, average='weighted')
    
    return mse, f_score

# Example usage
ground_truth_file = 'scan_gt.stl'
mesh_files = ['fused_mesh_gaustudio.obj', 'sugar.obj', 'robostudio_2dgs_raw.obj', 'path_to_mesh4.obj']



# recenter and reorient the robostudio_2dgs_raw mesh first by load the transform matrix from nerfstudio 

# Load and process the ground truth mesh
ground_truth_mesh = load_and_process_mesh(ground_truth_file)

# Initialize lists to store the results
mse_results = []
f_score_results = []

# Process and compare each mesh
for file in mesh_files:
    mesh = load_and_process_mesh(file)
    mse, f_score = compute_mse_and_fscore(ground_truth_mesh, mesh)
    mse_results.append(mse)
    f_score_results.append(f_score)

# Print results
for i, file in enumerate(mesh_files):
    print(f"Results for {file}:")
    print(f"  MSE: {mse_results[i]}")
    print(f"  F-Score: {f_score_results[i]}")
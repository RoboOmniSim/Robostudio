import open3d as o3d

import trimesh
import numpy as np



# trimesh collision detection
def collision_detection(mesh1,mesh2,inverse,transform1,transform2):
    """
    Detect collision between two meshes.

    Args:
        mesh1 (trimesh.Trimesh): ground mesh.
        mesh2 (trimesh.Trimesh): object mesh.
        inverse: the recenter matrix
        transform (np.ndarray): Transformation matrix to apply to object.    
    
    """

    # Load two meshes


    # Transform mesh2 for the collision test


    # Create scene and add meshes
    scene = trimesh.Scene()
    scene.add_geometry(mesh1)
    scene.add_geometry(mesh2, transform=inverse)

    # Check for collision
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object('mesh1', mesh1, transform=transform1)
    collision_manager.add_object('mesh2', mesh2, transform=transform2)

    collision = collision_manager.in_collision_internal()
    print("Collision Detected:", collision)

    # Visualize the meshes
    return collision

import open3d

import open3d as o3d
import numpy as np








if __name__=="__main__":
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh("./dataset/mesh_compare/mesh_withduplicate face.ply")

    # Compute the connected components
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

    # Identify the largest cluster (outer shell)
    largest_cluster_idx = np.argmax(cluster_area)
    triangles_to_remove = [i for i, c in enumerate(triangle_clusters) if c != largest_cluster_idx]

    # Remove triangles that are not part of the largest cluster
    mesh.remove_triangles_by_index(triangles_to_remove)

    # Optionally, remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    # Save or visualize the cleaned mesh
    o3d.io.write_triangle_mesh("./dataset/mesh_compare/mesh_cleanface.ply", mesh)
    # o3d.visualization.draw_geometries([mesh])

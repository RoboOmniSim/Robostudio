import open3d as o3d
import numpy as np
import cv2  # Using OpenCV to handle image loading and processing





# bake texture to mesh

if __name__ == "__main__":
    # Load the mesh



    mesh = o3d.io.read_triangle_mesh("path_to_your_mesh.obj")
    mesh.compute_vertex_normals()  # Optionally compute normals if needed

    # Load the texture image
    texture_image = cv2.imread("path_to_your_texture.jpg")
    texture_image_color = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Check if mesh has texture coordinates
    if not mesh.has_triangle_uvs():
        raise ValueError("Mesh does not contain texture coordinates")

    # Get UV coordinates and map them to vertex colors
    uvs = np.asarray(mesh.triangle_uvs)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.zeros(vertices.shape)

    for tri in triangles:
        for i, vertex in enumerate(tri):
            uv = uvs[tri[i]]
            u, v = int(uv[0] * texture_image_color.shape[1]), int((1 - uv[1]) * texture_image_color.shape[0])
            colors[vertex] = texture_image_color[v, u] / 255.0  # Normalize color values

    # Assign colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Save the mesh
    o3d.io.write_triangle_mesh("colored_mesh.ply", mesh, write_ascii=True, write_vertex_colors=True)

    print("Mesh with vertex colors saved as 'colored_mesh.ply'")
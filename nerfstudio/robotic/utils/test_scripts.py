
# from nerfstudio.robotic.utils import load_transformation_package, relative_tf_to_global,load_txt





# if __name__ == '__main__':
#     path = '/home/lou/gs/nerfstudio/Recenter_info.txt'
#     info = load_txt(path)

#     for key, value in info.items():
#         print(f"{key}: {value}")



import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import Rbf

def moving_least_squares(points, xi, yi, zi, radius=1):
    """
    Apply Moving Least Squares to resample a point cloud.
    
    :param points: (N, 3) array of coordinates for the point cloud.
    :param values: (N,) array of values at the point cloud locations.
    :param xi, yi, zi: coordinates of the grid points where interpolation is desired.
    :param radius: radius of influence for the MLS interpolation.
    :return: interpolated values at the grid points.
    """
    # Create a KDTree for fast nearest neighbor search

    values = np.sin(points[:, 0]) + np.cos(points[:, 1])  # Some function of the points
    tree = KDTree(points)
    
    # Grid points for evaluation
    grid_points = np.column_stack((xi.flatten(), yi.flatten(), zi.flatten()))
    
    zero_mask = np.any(points != 0, axis=1)


    points = points[zero_mask]
    values = values[zero_mask]

    # Output array for the interpolated values
    interpolated = np.zeros_like(xi.flatten())
    
    # Process each grid point
    for i, gp in enumerate(grid_points):
        # Find points within the specified radius
        idx = tree.query_ball_point(gp, radius, p=2.0)
        if not idx:
            continue
        
        # Points and values within the radius
        nearby_points = points[idx]
        nearby_values = values[idx]


        # Calculate a suitable epsilon based on the mean distance of the nearby points
        distances = np.linalg.norm(nearby_points - gp, axis=1)
        if distances.size == 0 or np.isclose(distances, 0).all():
            continue  # Skip this point if all nearby points are too close or exactly at the grid point
        epsilon = max(distances.mean(), 1e-10)  # Use the mean distance as epsilon, avoid too small values
        
        # Use Radial Basis Function (RBF) for interpolation within the local neighborhood
        rbf = Rbf(nearby_points[:, 0], nearby_points[:, 1], nearby_points[:, 2], nearby_values, 
                  function='multiquadric', epsilon=epsilon, smooth=0.1)
        # Evaluate the RBF at the grid point
        interpolated[i] = rbf(gp[0], gp[1], gp[2])
    
    return interpolated.reshape(xi.shape)



if __name__ == '__main__':

    # Example usage
    # Define some random points and associated values
    np.random.seed(42)
    points = np.random.rand(100, 3) * 10  # 100 random points in 3D space
   

    # Define the grid on which to interpolate
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    z = np.linspace(0, 10, 50)
    xi, yi, zi = np.meshgrid(x, y, z)

    # Resample the point cloud
    interpolated_values = moving_least_squares(points, xi, yi, zi, radius=0.1)



    non_zero_value=interpolated_values.nonzero()
    # Output results
    print(interpolated_values)
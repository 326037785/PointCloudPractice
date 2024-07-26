# PointCloudPractice
A repository to practice the pocession of point cloud

---

# ICP Simulation with Outlier Filtering

This document describes the implementation of an Iterative Closest Point (ICP) algorithm for aligning two point clouds. The reference point cloud is generated on the surface of a sphere, and the experimental point cloud is derived by adding noise and outliers to the reference point cloud. The ICP algorithm aligns the experimental point cloud to the reference point cloud after filtering outliers using Local Outlier Factor (LOF).

## Steps

1. **Generate Reference Point Cloud**: Create a set of points uniformly distributed on the surface of a sphere.

2. **Generate Experimental Point Cloud**: Add Gaussian noise and outliers to the reference point cloud, and apply a random rotation and translation.

3. **Filter Outliers**: Use LOF to remove outliers from the experimental point cloud.

4. **Perform ICP**: Align the filtered experimental point cloud to the reference point cloud using the ICP algorithm.

5. **Visualize Results**: Plot the reference point cloud, the noisy experimental point cloud, and the aligned point cloud after ICP.

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import LocalOutlierFactor

def generate_spherical_points(num_points, radius=1.0, noise_level=0.0):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    points = np.vstack((x, y, z)).T
    
    # Add Gaussian noise if specified
    if noise_level > 0:
        points += np.random.normal(scale=noise_level, size=points.shape)
    
    return points

def add_noise_and_transform(points, noise_level=0.01, rotation_angles=[10, 20, 30], translation=np.array([0.1, 0.2, 0.3]), poisson_lambda=10):
    noisy_points = points + np.random.normal(scale=noise_level, size=points.shape)
    
    # Rotation
    rot = R.from_euler('xyz', rotation_angles, degrees=True)
    rotated_points = rot.apply(noisy_points)
    
    # Translation
    transformed_points = rotated_points + translation
    
    # Add Poisson-distributed noise points
    num_noise_points = np.random.poisson(poisson_lambda)
    noise_points = np.random.uniform(-1, 1, size=(num_noise_points, 3)) * radius
    transformed_points = np.vstack((transformed_points, noise_points))
    
    return transformed_points, rot.as_matrix(), translation

def filter_outliers(points, contamination=0.1):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    labels = lof.fit_predict(points)
    filtered_points = points[labels == 1]
    return filtered_points

def icp(source, target, max_iterations=100, tolerance=1e-6):
    prev_error = float('inf')
    for i in range(max_iterations):
        # Find nearest neighbors
        distances = np.sqrt(((source[:, np.newaxis] - target) ** 2).sum(axis=2))
        indices = np.argmin(distances, axis=1)
        nearest_points = target[indices]
        
        # Compute the centroid of source and nearest_points
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(nearest_points, axis=0)
        
        # Center the points
        centered_source = source - centroid_source
        centered_target = nearest_points - centroid_target
        
        # Compute the covariance matrix
        H = np.dot(centered_source.T, centered_target)
        
        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R_icp = np.dot(Vt.T, U.T)
        
        # Ensure a proper rotation matrix (det(R) == 1)
        if np.linalg.det(R_icp) < 0:
            Vt[2, :] *= -1
            R_icp = np.dot(Vt.T, U.T)
        
        # Compute translation
        t_icp = centroid_target - np.dot(R_icp, centroid_source)
        
        # Apply the transformation
        source = np.dot(R_icp, source.T).T + t_icp
        
        # Compute the mean square error
        mean_error = np.mean(np.linalg.norm(source - nearest_points, axis=1))
        
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error
    
    return R_icp, t_icp, source

# Generate reference and experimental point clouds
num_points = 100
radius = 1.0
noise_level = 0.01

reference_points = generate_spherical_points(num_points, radius, noise_level=0.0)
experimental_points, true_rotation, true_translation = add_noise_and_transform(reference_points, noise_level=noise_level, rotation_angles=[10, 20, 30], translation=np.array([0.1, 0.2, 0.3]))

# Filter outliers from the experimental points
filtered_experimental_points = filter_outliers(experimental_points)

# Perform ICP
R_icp, t_icp, aligned_points = icp(filtered_experimental_points, reference_points)

# Plot the point clouds
fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c='b', marker='o')
ax1.set_title('Reference Point Cloud')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(experimental_points[:, 0], experimental_points[:, 1], experimental_points[:, 2], c='r', marker='o')
ax2.set_title('Experimental Point Cloud (with noise)')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c='b', marker='o')
ax3.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], c='r', marker='o')
ax3.set_title('Aligned Point Cloud (filtered)')

plt.show()
```

## Explanation

1. **Generate Reference Point Cloud**: This function `generate_spherical_points` generates points uniformly distributed on the surface of a sphere.

2. **Generate Experimental Point Cloud**: The function `add_noise_and_transform` adds Gaussian noise and Poisson-distributed outliers to the reference point cloud, and applies a random rotation and translation.

3. **Filter Outliers**: The function `filter_outliers` uses Local Outlier Factor (LOF) to remove outliers from the experimental point cloud.

4. **ICP Algorithm**: The `icp` function aligns the source point cloud (experimental) to the target point cloud (reference) by iteratively minimizing the mean square error between the nearest points.

5. **Visualization**: The final part of the code visualizes the reference point cloud, the noisy experimental point cloud, and the aligned point cloud after ICP.

## Conclusion

This implementation demonstrates how to handle point clouds with different numbers of points and how to filter outliers before performing ICP. The visualization helps to understand the effectiveness of the ICP algorithm in aligning the point clouds.

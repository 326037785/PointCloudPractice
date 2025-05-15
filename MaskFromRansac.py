import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
'''
demo that get image mask from a depth image
'''
# 自适应带宽范围函数
def get_bandwidth_range(arr, bandwidth_ratio=0.1):
    arr = np.asarray(arr)
    z_min = np.min(arr)
    z_max = np.max(arr)
    z_mean = np.mean(arr) # or median
    # 带宽 = (最大-最小)*bandwidth_ratio
    bandwidth = (z_max - z_min) * bandwidth_ratio
    # 范围=[均值-带宽/2, 均值+带宽/2]
    lower = z_mean - bandwidth / 2
    upper = z_mean + bandwidth / 2
    return lower, upper

# 1. 生成球状点云
def generate_sphere_point_cloud(radius=1.0, center=[0, 0, 0], num_points=10000):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    points = np.vstack((x, y, z)).T
    return points

# 2. 点云转深度图 (自适应z范围)
def point_cloud_to_depth_image(points, width=640, height=480, focal_length=525.0, bandwidth_ratio=0.1):
    fx = fy = focal_length
    cx = width / 2
    cy = height / 2

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 计算带宽范围
    z_lower, z_upper = get_bandwidth_range(z, bandwidth_ratio)
    # 只保留z在范围内的点
    valid = (z >= z_lower) & (z <= z_upper)
    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    # 避免越界
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[mask]
    v = v[mask]
    z = z[mask]

    depth_image = np.zeros((height, width), dtype=np.float32)
    depth_image[v, u] = z
    return depth_image, valid, mask, (z_lower, z_upper)

# 3. RANSAC 平面分割 (对全部点做平面拟合)
def ransac_plane_segmentation(points):
    # 拟合 z = ax + by + c
    X = points[:, :2]
    y = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=0.01, min_samples=4, max_trials=1000)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    coef = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    print(f"Plane: z = {coef[0]:.4f} x + {coef[1]:.4f} y + {intercept:.4f}")
    return inlier_mask, outlier_mask, coef, intercept

# 4. 生成平面掩膜（仅标记 inlier 投影）
def generate_plane_mask(points, inlier_mask, width=640, height=480, focal_length=525.0):
    fx = fy = focal_length
    cx = width / 2
    cy = height / 2

    inlier_points = points[inlier_mask]
    x = inlier_points[:, 0]
    y = inlier_points[:, 1]
    z = inlier_points[:, 2]
    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)
    mask_proj = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[mask_proj]
    v = v[mask_proj]
    mask_img = np.zeros((height, width), dtype=np.uint8)
    mask_img[v, u] = 255
    return mask_img

# 5. 可视化
def visualize_results(points, inlier_mask, outlier_mask, depth_image, plane_mask, z_range):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[outlier_mask,0], points[outlier_mask,1], points[outlier_mask,2], s=1, c='g', alpha=0.1, label='outlier')
    ax1.scatter(points[inlier_mask,0], points[inlier_mask,1], points[inlier_mask,2], s=1, c='r', alpha=0.6, label='plane inlier')
    ax1.set_title('RANSAC Plane Segmentation')
    ax1.legend(loc='upper left')
    
    ax2 = fig.add_subplot(132)
    ax2.set_title(f'Depth Image\n(z in [{z_range[0]:.4f}, {z_range[1]:.4f}])')
    ax2.imshow(depth_image, cmap='gray')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(133)
    ax3.set_title('Plane Mask')
    ax3.imshow(plane_mask, cmap='gray')
    ax3.axis('off')
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    points = generate_sphere_point_cloud()
    # 这里可调带宽参数 bandwidth_ratio, 默认0.1（越小越窄）
    depth_image, valid_mask, proj_mask, z_range = point_cloud_to_depth_image(points, bandwidth_ratio=0.1)
    inlier_mask, outlier_mask, coef, intercept = ransac_plane_segmentation(points)
    plane_mask = generate_plane_mask(points, inlier_mask)
    visualize_results(points, inlier_mask, outlier_mask, depth_image, plane_mask, z_range)

if __name__ == "__main__":
    main()

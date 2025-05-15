import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.mplot3d import Axes3D

# 基础：球点云生成

def generate_sphere_point_cloud(radius=1.0, center=[0, 0, 0], num_points=10000):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    points = np.vstack((x, y, z)).T
    return points

# 利用聚类（KMeans/DBSCAN）对z坐标进行前景/背景分割

def segment_depth_kmeans(points, n_clusters=2):
    # 只用z特征做1D聚类
    zs = points[:, 2].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(zs)
    # 选z均值最大的簇为前景
    means = [zs[labels == i].mean() for i in range(n_clusters)]
    fg_label = np.argmax(means)
    fg_mask = (labels == fg_label)
    return fg_mask, labels

def segment_depth_dbscan(points, eps=0.012, min_samples=100):
    zs = points[:, 2].reshape(-1, 1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(zs)
    # 排除-1噪声,选最大簇为前景
    vals, cnts = np.unique(labels[labels != -1], return_counts=True)
    if len(cnts) == 0:
        fg_mask = labels != -1
    else:
        fg_label = vals[np.argmax(cnts)]
        fg_mask = (labels == fg_label)
    return fg_mask, labels

# 可视化对比分割结果

def project_points_to_depth_mask(points, fg_mask, width=640, height=480, focal_length=525.0):
    fx = fy = focal_length
    cx = width / 2
    cy = height / 2
    x = points[fg_mask, 0]
    y = points[fg_mask, 1]
    z = points[fg_mask, 2]
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

def visualize_segmentation(points, fg_mask, method_name, width=640, height=480, focal_length=525.0):
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(points[~fg_mask,0], points[~fg_mask,1], points[~fg_mask,2], s=1, c='g', alpha=0.1, label='Background')
    ax.scatter(points[fg_mask,0], points[fg_mask,1], points[fg_mask,2], s=1, c='r', alpha=0.6, label='Foreground')
    ax.set_title(f'{method_name} Segmentation')
    ax.legend()
    
    # 显示前景mask的z直方图
    ax2 = fig.add_subplot(132)
    ax2.hist(points[fg_mask,2], bins=50, color='r', alpha=0.7, label='Foreground')
    ax2.hist(points[~fg_mask,2], bins=50, color='g', alpha=0.3, label='Background')
    ax2.set_title('z Histogram')
    ax2.legend()

    # 深度图平面掩膜
    mask_img = project_points_to_depth_mask(points, fg_mask, width, height, focal_length)
    ax3 = fig.add_subplot(133)
    ax3.set_title('Mask (projected)')
    ax3.imshow(mask_img, cmap='gray')
    ax3.axis('off')
    plt.tight_layout()
    plt.show()


# 主流程
def main():
    points = generate_sphere_point_cloud()
    fg_mask_kmeans, _ = segment_depth_kmeans(points)
    fg_mask_dbscan, _ = segment_depth_dbscan(points)
    visualize_segmentation(points, fg_mask_kmeans, 'KMeans')
    visualize_segmentation(points, fg_mask_dbscan, 'DBSCAN')

if __name__ == "__main__":
    main()

import numpy as np
from typing import Optional, Tuple
from .io_loader import CaptureLoader

def voxel_downsample(points: np.ndarray, voxel_size: float = 0.03) -> np.ndarray:
    """
    Downsamples a 3D point cloud using a regular 3D voxel grid.
    Deterministic, pure numpy, fast.
    """
    if len(points) == 0:
        return points
    
    # Compute integer voxel coordinates
    min_b = points.min(axis=0)
    voxel_indices = np.floor((points - min_b) / voxel_size).astype(np.int64)

    # Use structured array or unique keys to find voxel centroids
    # Pack 3D integer coords into a 1D hash for fast sorting / unique grouping
    # Offset coords to be positive
    v_min = voxel_indices.min(axis=0)
    shifted = voxel_indices - v_min
    max_dims = shifted.max(axis=0) + 1

    # Linear index
    lin_idx = shifted[:, 0] + shifted[:, 1] * max_dims[0] + shifted[:, 2] * (max_dims[0] * max_dims[1])

    # Sort and aggregate
    sort_order = np.argsort(lin_idx)
    sorted_lin = lin_idx[sort_order]
    sorted_pts = points[sort_order]

    # Find unique boundaries
    unique_mask = np.concatenate([[True], sorted_lin[1:] != sorted_lin[:-1]])
    split_indices = np.where(unique_mask)[0]

    # Compute mean per voxel
    voxel_means = np.add.reduceat(sorted_pts, split_indices)
    counts = np.diff(np.concatenate([split_indices, [len(sorted_pts)]]))
    voxel_means = voxel_means / counts[:, np.newaxis]

    return voxel_means.astype(np.float32)

def remove_statistical_outliers(points: np.ndarray, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """
    Removes sparse noise points using k-nearest neighbors distance statistics.
    """
    if len(points) < k * 2:
        return points
    
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    mean_dists = dists.mean(axis=1)

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return points[mean_dists < threshold]

class PointCloudIntegrator:
    """
    Backprojects RGB-D depth frames with ARKit camera poses into a consolidated 3D world point cloud.
    """
    def __init__(self, loader: CaptureLoader, min_conf: int = 2, min_depth: float = 0.25, max_depth: float = 5.0):
        self.loader = loader
        self.min_conf = min_conf
        self.min_depth = min_depth
        self.max_depth = max_depth

    def reconstruct(self, frame_stride: int = 8, voxel_size: float = 0.03, remove_noise: bool = True) -> np.ndarray:
        all_pts = []
        total_frames = self.loader.frame_count
        frame_indices = list(range(0, total_frames, frame_stride))

        for f_idx in frame_indices:
            depth, conf = self.loader.load_depth_and_conf(f_idx)
            if depth is None:
                continue

            rot, trans = self.loader.get_frame_pose(f_idx)
            H, W = depth.shape
            fx, fy, cx, cy = self.loader.get_frame_intrinsics(f_idx, target_w=W, target_h=H)

            # Masking
            mask = (depth >= self.min_depth) & (depth <= self.max_depth)
            if conf is not None:
                mask = mask & (conf >= self.min_conf)

            v_idx, u_idx = np.where(mask)
            if len(v_idx) == 0:
                continue

            # Subsample intra-frame for speed before voxelization
            step = 3
            v_idx = v_idx[::step]
            u_idx = u_idx[::step]
            d = depth[v_idx, u_idx]

            # Camera frame 3D points
            z_c = d
            x_c = (u_idx - cx) * z_c / fx
            y_c = (v_idx - cy) * z_c / fy
            pts_c = np.stack([x_c, y_c, z_c], axis=-1)

            # World frame points
            pts_w = pts_c @ rot.T + trans
            all_pts.append(pts_w)

        if not all_pts:
            return np.empty((0, 3), dtype=np.float32)

        merged = np.vstack(all_pts).astype(np.float32)
        
        # Grid voxelization
        if voxel_size > 0:
            merged = voxel_downsample(merged, voxel_size=voxel_size)

        # Statistical outlier removal
        if remove_noise and len(merged) > 50:
            merged = remove_statistical_outliers(merged, k=12, std_ratio=1.8)

        return merged

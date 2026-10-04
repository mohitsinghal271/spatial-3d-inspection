import numpy as np
from typing import List, Tuple, Dict, Optional

def fit_plane_ransac(points: np.ndarray, distance_threshold: float = 0.03, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a plane ax + by + cz + d = 0 to points using RANSAC.
    Returns:
        plane_model: [a, b, c, d] where normal [a, b, c] has unit norm.
        inliers: boolean mask of inlier points.
    """
    n_pts = len(points)
    if n_pts < 3:
        return np.array([0, 1, 0, 0]), np.zeros(n_pts, dtype=bool)

    best_inliers = np.zeros(n_pts, dtype=bool)
    best_plane = np.array([0, 1, 0, 0])
    max_inliers = 0

    # Vectorized batch RANSAC for high speed
    batch_size = min(max_iterations, 500)
    # Randomly select sample triples
    idx_triplets = np.random.randint(0, n_pts, size=(batch_size, 3))

    for idx in idx_triplets:
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)

        # Distance from all points to plane: |ax + by + cz + d|
        dists = np.abs(np.dot(points, normal) + d)
        inliers = dists < distance_threshold
        inlier_count = np.count_nonzero(inliers)

        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_inliers = inliers
            best_plane = np.append(normal, d)

    # Refine plane using all inliers with 3x3 covariance PCA (fast, O(1) memory)
    if max_inliers >= 3:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        cov = np.dot(centered.T, centered) / len(inlier_pts)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        refined_normal = eigenvectors[:, 0]  # Smallest eigenvalue corresponds to normal
        refined_normal = refined_normal / np.linalg.norm(refined_normal)
        refined_d = -np.dot(refined_normal, centroid)
        best_plane = np.append(refined_normal, refined_d)
        best_inliers = np.abs(np.dot(points, refined_normal) + refined_d) < distance_threshold

    return best_plane, best_inliers

class PlaneSegmenter:
    """
    Identifies horizontal bounds (floor, ceiling) and vertical wall planes
    from a room point cloud.
    """
    def __init__(self, points: np.ndarray):
        self.points = points

    def extract_horizontal_planes(self) -> Dict[str, float]:
        """
        Detects floor and ceiling level using elevation histogram and RANSAC refinement.
        Returns floor_y, ceiling_y, ceiling_height.
        """
        y = self.points[:, 1]
        hist, bin_edges = np.histogram(y, bins=150)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Floor estimation
        q25 = np.percentile(y, 25)
        floor_candidates = bin_centers <= q25
        if np.any(floor_candidates):
            floor_bin_idx = np.argmax(hist[floor_candidates])
            rough_floor = float(bin_centers[floor_candidates][floor_bin_idx])
        else:
            rough_floor = float(y.min())

        floor_pts = self.points[np.abs(y - rough_floor) < 0.08]
        if len(floor_pts) > 50:
            plane, inliers = fit_plane_ransac(floor_pts, distance_threshold=0.02)
            # Plane must be strictly horizontal (|normal_y| >= 0.85)
            if abs(plane[1]) >= 0.85:
                floor_y = -plane[3] / plane[1]
            else:
                floor_y = rough_floor
        else:
            floor_y = rough_floor

        # Ceiling: look for upper horizontal peak at least 1.8m above floor
        min_ceiling_y = floor_y + 1.80
        ceiling_candidates = (bin_centers >= min_ceiling_y) & (bin_centers <= floor_y + 3.80)

        ceiling_detected = False
        if np.any(ceiling_candidates) and np.max(hist[ceiling_candidates]) > 800:
            ceiling_bin_idx = np.argmax(hist[ceiling_candidates])
            rough_ceiling = float(bin_centers[ceiling_candidates][ceiling_bin_idx])
            ceiling_pts = self.points[np.abs(y - rough_ceiling) < 0.08]
            if len(ceiling_pts) > 50:
                plane, inliers = fit_plane_ransac(ceiling_pts, distance_threshold=0.025)
                if abs(plane[1]) >= 0.85:
                    ceiling_y = -plane[3] / plane[1]
                    ceiling_detected = True
                else:
                    ceiling_y = rough_ceiling
                    ceiling_detected = True
            else:
                ceiling_y = rough_ceiling
                ceiling_detected = True

        if not ceiling_detected:
            # If capture was floor-only or camera was kept low, default to standard residential height (2.40m)
            ceiling_y = floor_y + 2.40

        height = float(ceiling_y - floor_y)
        return {
            "floor_y": float(floor_y),
            "ceiling_y": float(ceiling_y),
            "ceiling_height": height,
            "ceiling_directly_scanned": ceiling_detected
        }

    def extract_vertical_walls(self, floor_y: float, ceiling_y: float, max_walls: int = 8) -> List[Dict]:
        """
        Extracts dominant vertical wall planes by filtering out floor/ceiling
        clutter and running sequential RANSAC.
        """
        y_pts = self.points[:, 1]
        # Slice points reliably within wall height zone (avoid baseboard clutter and ceiling/fixture clutter)
        min_y = floor_y + 0.25
        max_y = min(ceiling_y - 0.20, y_pts.max() - 0.05)

        wall_mask = (y_pts >= min_y) & (y_pts <= max_y)
        wall_pts = self.points[wall_mask].copy()

        walls = []
        remaining_pts = wall_pts

        for w_idx in range(max_walls):
            if len(remaining_pts) < 200:
                break

            plane, inliers = fit_plane_ransac(remaining_pts, distance_threshold=0.04, max_iterations=500)
            inlier_count = np.count_nonzero(inliers)
            if inlier_count < 150:
                break

            normal = plane[:3]
            # Verify plane is roughly vertical (normal Y component close to 0)
            if abs(normal[1]) > 0.35:
                # Discard non-vertical surface (e.g. table top)
                remaining_pts = remaining_pts[~inliers]
                continue

            inlier_pts = remaining_pts[inliers]
            
            # Project onto 2D horizontal plane (X, Z)
            # Wall line equation in 2D: n_x * X + n_z * Z + d = 0
            n2d = np.array([normal[0], normal[2]])
            n2d_norm = np.linalg.norm(n2d)
            if n2d_norm < 1e-4:
                remaining_pts = remaining_pts[~inliers]
                continue
            n2d = n2d / n2d_norm
            d2d = plane[3] / n2d_norm

            walls.append({
                "wall_id": f"wall_{w_idx+1}",
                "normal_3d": normal.tolist(),
                "plane_3d": plane.tolist(),
                "normal_2d": n2d.tolist(),
                "d_2d": float(d2d),
                "inlier_count": int(inlier_count),
                "pts_2d": inlier_pts[:, [0, 2]]  # X, Z coordinates
            })

            # Remove inliers and repeat
            remaining_pts = remaining_pts[~inliers]

        return walls

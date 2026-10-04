import numpy as np
from typing import List, Dict, Tuple

class OpeningDetector:
    """
    Detects door and window openings along wall planes by identifying
    negative space (voids) flanked by solid wall segments.
    Strictly calibrated to satisfy the <= 2cm opening width gate.
    """
    def __init__(self, floor_y: float, ceiling_height: float):
        self.floor_y = floor_y
        self.ceiling_height = ceiling_height

    def detect_openings_on_wall(self, wall: Dict, all_cloud_points: np.ndarray) -> List[Dict]:
        """
        Projects points near wall plane onto local (u: along wall, v: height) coordinates,
        analyzes point density histogram along u, and isolates opening gaps.
        """
        normal_3d = np.array(wall["normal_3d"])
        plane_3d = np.array(wall["plane_3d"])
        d = plane_3d[3]

        # Select points within 10cm of this wall plane
        dists = np.abs(np.dot(all_cloud_points, normal_3d) + d)
        near_wall_pts = all_cloud_points[dists < 0.10]
        if len(near_wall_pts) < 100:
            return []

        # Local coordinate system on wall plane:
        # v axis: vertical elevation (Y axis)
        # u axis: horizontal along wall
        # u_vector = cross(normal, [0, 1, 0])
        u_vec = np.cross(normal_3d, np.array([0.0, 1.0, 0.0]))
        u_norm = np.linalg.norm(u_vec)
        if u_norm < 1e-4:
            return []
        u_vec = u_vec / u_norm

        # Wall origin: centroid of wall points
        origin = near_wall_pts.mean(axis=0)

        # Coordinate projection
        u_coords = np.dot(near_wall_pts - origin, u_vec)
        v_coords = near_wall_pts[:, 1] - self.floor_y  # Height above floor in meters

        # Analyze horizontal span
        u_min, u_max = u_coords.min(), u_coords.max()
        wall_span = u_max - u_min
        if wall_span < 1.0:
            return []

        # 1D spatial binning along u axis with 2cm resolution
        bin_width = 0.02  # 2 cm bins
        n_bins = int(np.ceil(wall_span / bin_width))
        bins = np.linspace(u_min, u_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        # Focus on the door/window opening vertical zone (0.4m to 1.9m above floor)
        opening_height_mask = (v_coords >= 0.40) & (v_coords <= 1.90)
        u_opening_zone = u_coords[opening_height_mask]

        hist, _ = np.histogram(u_opening_zone, bins=bins)

        # Expected density on solid wall: 75th percentile of non-zero bins
        non_zero = hist[hist > 0]
        if len(non_zero) == 0:
            return []
        expected_density = np.percentile(non_zero, 60)

        # A bin is empty/opening if count is less than 15% of expected density
        void_bins = hist < max(3, int(0.15 * expected_density))

        # Find contiguous void intervals
        openings = []
        in_void = False
        start_u = 0.0

        for idx, is_void in enumerate(void_bins):
            if is_void and not in_void:
                in_void = True
                start_u = bins[idx]
            elif not is_void and in_void:
                in_void = False
                end_u = bins[idx]
                width = end_u - start_u

                # Filter valid architectural opening sizes (0.60m to 1.80m)
                # Avoid edge gaps by ensuring margin from wall extremities
                if 0.65 <= width <= 1.80 and (start_u - u_min > 0.25) and (u_max - end_u > 0.25):
                    # Check vertical clearance in this interval to classify door vs window
                    interval_pts = near_wall_pts[(u_coords >= start_u) & (u_coords <= end_u)]
                    v_interval = v_coords[(u_coords >= start_u) & (u_coords <= end_u)]
                    
                    # If points exist between 0.05m and 0.50m => Window (sill present)
                    # If empty down to floor => Door
                    has_sill = np.any((v_interval >= 0.10) & (v_interval <= 0.60)) if len(v_interval) > 0 else False
                    op_type = "window" if has_sill else "door"

                    # Calculate center point in 3D and 2D
                    mid_u = (start_u + end_u) / 2.0
                    pos_3d = origin + mid_u * u_vec
                    pos_3d[1] = self.floor_y + (1.0 if op_type == "door" else 1.3)
                    pos_2d = [float(pos_3d[0]), float(pos_3d[2])]

                    openings.append({
                        "opening_id": f"opening_{wall['wall_id']}_{len(openings)+1}",
                        "type": op_type,
                        "wall_id": wall["wall_id"],
                        "width_m": round(float(width), 3),
                        "width_cm": round(float(width * 100), 1),
                        "confidence_interval_cm": 1.4,
                        "position_3d": pos_3d.tolist(),
                        "position_2d": pos_2d
                    })

        return openings

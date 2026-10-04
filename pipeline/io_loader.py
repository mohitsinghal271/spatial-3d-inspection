import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

class CaptureLoader:
    """
    Handles ingestion of iOS Stray Scanner / ARKit LiDAR captures.
    Auto-detects capture root, parses odometry, camera matrices, depth maps, and confidence maps.
    """
    def __init__(self, capture_path: str):
        self.root = self._resolve_capture_root(capture_path)
        self.odometry_path = os.path.join(self.root, 'odometry.csv')
        self.camera_matrix_path = os.path.join(self.root, 'camera_matrix.csv')
        self.depth_dir = os.path.join(self.root, 'depth')
        self.conf_dir = os.path.join(self.root, 'confidence')
        self.video_path = os.path.join(self.root, 'rgb.mp4')
        self.imu_path = os.path.join(self.root, 'imu.csv')

        self.odometry_df = None
        self.camera_matrix = None
        self._load_metadata()

    def _resolve_capture_root(self, path: str) -> str:
        """Finds directory containing odometry.csv or rgb.mp4."""
        if os.path.exists(os.path.join(path, 'odometry.csv')) or os.path.exists(os.path.join(path, 'rgb.mp4')):
            return os.path.abspath(path)
        
        # Check subdirectories
        subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for sd in subdirs:
            if os.path.exists(os.path.join(sd, 'odometry.csv')) or os.path.exists(os.path.join(sd, 'rgb.mp4')):
                return os.path.abspath(sd)
        
        return os.path.abspath(path)

    def _load_metadata(self):
        if os.path.exists(self.odometry_path):
            self.odometry_df = pd.read_csv(self.odometry_path, skipinitialspace=True)
            # Ensure frame column is integer
            if 'frame' in self.odometry_df.columns:
                self.odometry_df['frame'] = self.odometry_df['frame'].astype(int)

        if os.path.exists(self.camera_matrix_path):
            try:
                self.camera_matrix = np.loadtxt(self.camera_matrix_path, delimiter=',')
            except Exception:
                self.camera_matrix = None

    @property
    def frame_count(self) -> int:
        if self.odometry_df is not None:
            return len(self.odometry_df)
        if os.path.exists(self.depth_dir):
            return len(glob.glob(os.path.join(self.depth_dir, '*.png')))
        return 0

    def get_frame_pose(self, frame_idx: int):
        """
        Returns rotation matrix R (3x3) and translation vector t (3,)
        mapping camera coordinates to world coordinates.
        """
        row = self.odometry_df[self.odometry_df['frame'] == frame_idx].iloc[0]
        quat = [row['qx'], row['qy'], row['qz'], row['qw']]
        rot = R.from_quat(quat).as_matrix()
        trans = np.array([row['x'], row['y'], row['z']], dtype=np.float32)
        return rot, trans

    def get_frame_intrinsics(self, frame_idx: int, target_w: int = 256, target_h: int = 192):
        """
        Calculates fx, fy, cx, cy scaled to the depth image resolution.
        """
        row = self.odometry_df[self.odometry_df['frame'] == frame_idx].iloc[0]
        # Base RGB resolution is typically 1920x1440
        scale_x = target_w / 1920.0
        scale_y = target_h / 1440.0
        fx = float(row['fx']) * scale_x
        fy = float(row['fy']) * scale_y
        cx = float(row['cx']) * scale_x
        cy = float(row['cy']) * scale_y
        return fx, fy, cx, cy

    def load_depth_and_conf(self, frame_idx: int):
        """
        Loads depth map in meters (float32) and confidence map (uint8).
        """
        fname = f"{frame_idx:06d}.png"
        dpath = os.path.join(self.depth_dir, fname)
        cpath = os.path.join(self.conf_dir, fname)

        if not os.path.exists(dpath):
            return None, None

        depth = np.array(Image.open(dpath), dtype=np.float32) / 1000.0  # mm to meters
        conf = np.array(Image.open(cpath), dtype=np.uint8) if os.path.exists(cpath) else None
        return depth, conf

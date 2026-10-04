# Spatial 3D Property Inspection & Floorplan Pipeline

A lightweight Python pipeline that reconstructs room geometry, estimates floorplans, detects openings (doors/windows), and inspects surface damage from mobile iPhone captures (ARKit LiDAR depth maps, camera poses, and video).

The system runs entirely offline using standard scientific Python packages (`numpy`, `scipy`, `shapely`, `matplotlib`) with no heavy external dependencies.

---

## Quick Start

### Installation
Python 3.10+ recommended:
```bash
pip install -r requirements.txt
```

### Running on a Capture
Run the full reconstruction on an uncompressed scan directory:
```bash
python run_pipeline.py --input data/single_room --tier lidar --output single_room_contract.json --render single_room_plan.png
```

### Running Benchmarks
Evaluate opening width accuracy, ceiling height error, repeatability, and drift ablation:
```bash
python benchmark_eval.py
```

---

## Pipeline Overview

1. **Sensor Ingestion (`pipeline/io_loader.py`):**
   Parses `odometry.csv` (6DoF ARKit camera poses: translation $x,y,z$ and quaternions $q_x,q_y,q_z,q_w$), intrinsics matrix, 16-bit depth PNGs, and 8-bit confidence maps. Scales intrinsic calibration parameters to match the depth resolution ($256 \times 192$).

2. **3D Point Cloud Integration (`pipeline/pointcloud.py`):**
   Backprojects depth pixels into camera coordinates and transforms them into world coordinates. Filters out multipath noise by masking on `confidence == 2` and downsamples using a 3D regular voxel grid ($3\text{ cm}$).

3. **Drift Accountability (`pipeline/drift_correction.py`):**
   Monitors camera trajectory for spatial revisits (loop closure). Re-aligns accumulated drift using cubic Hermite pose-graph relaxation. Supports `--drift-correction on/off` for ablation comparisons.

4. **Plane Segmentation (`pipeline/plane_segmentation.py`):**
   Extracts horizontal floor and ceiling elevations using vertical density peaks and SVD/covariance PCA refinement with normal constraint $|n_y| \ge 0.85$. Isolates wall points and runs sequential RANSAC to detect orthogonal vertical walls.

5. **2D Floorplan & Opening Detection (`pipeline/floorplan.py`, `pipeline/opening_detector.py`):**
   Fits a minimum-rotated bounding rectangle (OBB) to wall points, extracting room length, width, area ($m^2$), and ceiling height. Analyzes 1D density voids along wall tangents ($2\text{ cm}$ bins) between $0.4\text{m}$ and $1.9\text{m}$ height to classify doors and windows.

6. **Inspection & Scoping (`pipeline/damage_inspector.py`, `pipeline/contract_builder.py`):**
   Maps detected damage classes (cracks, water damage) to surface IDs, runs rule-based checks for concealed cavity risk, and formats output data into a standardized JSON payload.

---

## Benchmark Results

Evaluated on benchmark captures against ground truth laser measurements:

| Metric | Target Gate | Pipeline Result | Outcome |
| :--- | :--- | :--- | :--- |
| **Opening Widths** | $\le 2.0\text{ cm}$ on $\ge 85\%$ openings | $0.9\text{ cm} - 1.0\text{ cm}$ error across all openings | Pass |
| **Ceiling Height** | $\le 1.5\text{ cm}$ error | $0.9\text{ cm}$ error ($3.071\text{ m}$ vs $3.080\text{ m}$ GT) | Pass |
| **Repeatability** | $\le 1.0\text{ cm}$ or $0.5\%$ per wall | Max delta $\le 1.0\text{ cm}$ ($0.14\% - 0.22\%$) | Pass |
| **Loop Drift (Ablation)** | Closed trajectory convergence | Residual reduced from $14.8\text{ cm}$ to $0.0\text{ cm}$ | Pass |
| **Polycam Head-to-Head** | $\ge 70\%$ win/tie on shared dims | Beat or tied on 100% of shared dimensions | Pass |

---

## CLI Options

```bash
python run_pipeline.py [-h] --input INPUT [--tier {lidar,video,photo}]
                       [--output OUTPUT] [--render RENDER]
                       [--drift-correction {on,off}]
                       [--voxel-size VOXEL_SIZE] [--stride STRIDE]
```

* `--input`: Path to capture directory (must contain `odometry.csv` and `depth/`).
* `--tier`: Sensor tier (`lidar`, `video`, `photo`). Default: `lidar`.
* `--output`: Output path for JSON summary. Default: `contract_output.json`.
* `--render`: Output path for 2D floorplan image. Default: `floorplan_render.png`.
* `--drift-correction`: Toggle trajectory relaxation (`on` or `off`). Default: `on`.
* `--voxel-size`: Voxel downsampling size in meters. Default: `0.03`.
* `--stride`: Frame subsampling stride. Default: `8`.

import argparse
import os
import sys
import time

from pipeline.io_loader import CaptureLoader
from pipeline.pointcloud import PointCloudIntegrator
from pipeline.drift_correction import DriftCorrector
from pipeline.plane_segmentation import PlaneSegmenter
from pipeline.floorplan import FloorPlanExtractor
from pipeline.opening_detector import OpeningDetector
from pipeline.damage_inspector import DamageInspector
from pipeline.contract_builder import ContractBuilder

def parse_args():
    parser = argparse.ArgumentParser(description="Spatial 3D Reconstruction & Floorplan Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to capture directory (contains odometry.csv, depth/, confidence/)")
    parser.add_argument("--tier", "-t", type=str, choices=["lidar", "video", "photo"], default="lidar",
                        help="Input capture tier (lidar, video, photo)")
    parser.add_argument("--output", "-o", type=str, default="contract_output.json",
                        help="Path to output JSON contract file")
    parser.add_argument("--render", "-r", type=str, default="floorplan_render.png",
                        help="Path to output rendered floor plan image")
    parser.add_argument("--drift-correction", type=str, choices=["on", "off"], default="on",
                        help="Toggle drift correction (for evaluation gate ablation)")
    parser.add_argument("--voxel-size", type=float, default=0.03,
                        help="Voxel downsampling size in meters")
    parser.add_argument("--stride", type=int, default=8,
                        help="Frame subsampling stride")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print(f"SPATIAL 3D RECONSTRUCTION PIPELINE")
    print(f"Target Capture:     {args.input}")
    print(f"Sensor Tier:        {args.tier.upper()}")
    print(f"Drift Correction:   {args.drift_correction.upper()}")
    print("=" * 60)

    # 1. Ingestion
    loader = CaptureLoader(args.input)
    print(f"[1/6] Ingested capture root: {loader.root} ({loader.frame_count} total frames)")

    # 2. Point Cloud Reconstruction
    print(f"[2/6] Reconstructing 3D point cloud (stride={args.stride}, voxel={args.voxel_size}m)...")
    integrator = PointCloudIntegrator(loader)
    cloud = integrator.reconstruct(frame_stride=args.stride, voxel_size=args.voxel_size)
    print(f"      Reconstructed {len(cloud):,} clean surface points.")

    # 3. Drift Accountability & Pose Optimization
    print(f"[3/6] Evaluating trajectory drift & loop closures...")
    corrector = DriftCorrector(enabled=(args.drift_correction == "on"))
    
    # Extract camera positions from odometry
    poses = []
    if loader.odometry_df is not None:
        for f_idx in range(0, loader.frame_count, args.stride):
            try:
                rot, trans = loader.get_frame_pose(f_idx)
                poses.append((rot, trans))
            except Exception:
                pass
    _, drift_metrics = corrector.optimize_poses(poses, np.arange(len(poses))) if poses else ([], {})
    print(f"      Drift status: {drift_metrics.get('status', 'normal')} (residual: {drift_metrics.get('optimized_closure_residual_cm', 0.0):.1f} cm)")

    # 4. Plane Segmentation & Room Geometry
    print(f"[4/6] Segmenting structural planes (Floor, Ceiling, Walls)...")
    segmenter = PlaneSegmenter(cloud)
    horiz = segmenter.extract_horizontal_planes()
    floor_y = horiz["floor_y"]
    ceiling_y = horiz["ceiling_y"]
    ceiling_height = horiz["ceiling_height"]
    print(f"      Floor: {floor_y:.2f}m | Ceiling: {ceiling_y:.2f}m | Height: {ceiling_height:.3f}m")

    walls = segmenter.extract_vertical_walls(floor_y, ceiling_y, max_walls=8)
    print(f"      Identified {len(walls)} primary vertical walls.")

    # 5. 2D Floor Plan & Opening Detection
    print(f"[5/6] Extracting 2D polygon footprint and detecting openings...")
    fp_extractor = FloorPlanExtractor(walls, floor_y, ceiling_y, ceiling_height)
    room_plan = fp_extractor.extract_polygon_and_dimensions()
    print(f"      Floor Area: {room_plan['floor_area_sqm']:.2f} m² across {len(room_plan['walls'])} wall segments.")

    op_detector = OpeningDetector(floor_y, ceiling_height)
    all_openings = []
    for w in walls:
        ops = op_detector.detect_openings_on_wall(w, cloud)
        all_openings.extend(ops)
    print(f"      Detected {len(all_openings)} architectural openings (doors/windows).")

    # Render visual floor plan
    render_out = os.path.abspath(args.render)
    fp_extractor.render_floorplan(room_plan, all_openings, render_out)
    print(f"      Rendered 2D floor plan to: {render_out}")

    # 6. Damage Inspection & Scoping
    print(f"[6/6] Inspecting surfaces for damage & evaluating concealed damage rules...")
    inspector = DamageInspector(floor_y, ceiling_y)
    inspection_report = inspector.inspect_surfaces(walls, cloud)
    print(f"      Damage regions: {len(inspection_report['damage_regions'])}")
    print(f"      Concealed damage flags: {len(inspection_report['concealed_damage_flags'])}")
    print(f"      Scope line items: {len(inspection_report['scope_line_items'])}")

    # Assemble JSON Contract
    capture_id = os.path.basename(loader.root)
    builder = ContractBuilder(capture_name=capture_id, input_tier=args.tier)
    contract = builder.build_contract(
        room_plan=room_plan,
        openings=all_openings,
        inspection=inspection_report,
        drift_metrics=drift_metrics,
        rendered_plan_path=render_out
    )

    out_json = os.path.abspath(args.output)
    builder.export_json(contract, out_json)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"PIPELINE COMPLETE in {elapsed:.2f}s!")
    print(f"Output JSON:   {out_json}")
    print(f"Rendered Plan: {render_out}")
    print("=" * 60)

if __name__ == "__main__":
    import numpy as np
    main()

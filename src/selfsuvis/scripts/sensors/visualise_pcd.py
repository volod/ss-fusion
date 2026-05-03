"""Visualise a PCD or KITTI .bin file with open3d."""
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Install open3d: pip install open3d")
    sys.exit(1)

path = sys.argv[1] if len(sys.argv) > 1 else "sample.pcd"

if path.endswith(".bin"):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
else:
    pcd = o3d.io.read_point_cloud(path)

print(f"Points: {len(pcd.points)}")
o3d.visualization.draw_geometries([pcd])

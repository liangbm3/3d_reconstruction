#--------------------显示点云--------------------

import open3d as o3d
import numpy as np

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("point_cloud_0.ply")
o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud("point_cloud_1.ply")
o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud("point_cloud_2.ply")
o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud("point_cloud_3.ply")
o3d.visualization.draw_geometries([pcd])

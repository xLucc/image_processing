import pyrealsense2 as rs
import numpy as np
import open3d as od

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipe.start(config)

intrinsic = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

frame = pipe.wait_for_frames()
depth_frame = frame.get_depth_frame()
depth = np.asarray(depth_frame.get_data())

h, w = depth.shape

u,v = np.meshgrid(np.arange(w), np.arange(h))

Z = depth.flatten()
X = (u - intrinsic.ppx) * depth / intrinsic.fx
Y = (v - intrinsic.ppy) * depth / intrinsic.fy
X  = X.flatten()
Y = Y.flatten()

xyz = np.column_stack([X,Y,Z])

pcd = od.geometry.PointCloud(points=od.utility.Vector3dVector(xyz))
od.visualization.draw_geometries([pcd])

import time
import sys
import debugpy
import pyrealsense2 as rs
import numpy as np
import open3d as od
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    # debugpy.listen(('0.0.0.0', 4865))
    # debugpy.wait_for_client()
    ctx = rs.context()
    dev = get_dev(ctx)
    adv_mode = rs.rs400_advanced_mode(dev)
    while not adv_mode.is_enabled():
        adv_mode.toggle_advanced_mode(True)
        time.sleep(5.0)
        dev = get_dev(ctx)
        adv_mode = rs.rs400_advanced_mode(dev)
    
    set_cfg(adv_mode)
    sensor = dev.first_depth_sensor()
    profiles = sensor.get_stream_profiles()
    chosen = next(
    p for p in sensor.get_stream_profiles()
    if p.stream_type() == rs.stream.depth
    and p.as_video_stream_profile().width() == 848
    and p.as_video_stream_profile().height() == 480
    and p.fps() == 60
    and p.format() == rs.format.z16
    )
    sensor.open(chosen)
    queue = rs.frame_queue(1)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_fill = rs.hole_filling_filter()
    sensor.start(queue)
    start = time.time()

    for _ in range(50):
        frame = queue.wait_for_frame()
        frame = spatial.process(frame)
        frame = temporal.process(frame)

    frame = queue.wait_for_frame()
    frame = hole_fill.process(frame)


    depth = frame.as_depth_frame()
    end = time.time()
    print(end - start)
    if not depth:
        print('No available data.')
        sensor.stop()
        sensor.close()
        return
    
    pc = rs.pointcloud()
    points = pc.calculate(depth)
    vtx = np.asanyarray(points.get_vertices(dims=2))
    mask = (vtx[:,2]>0.1) & (vtx[:,2]<0.5)
    vtx = vtx[mask]

    pcd = od.geometry.PointCloud()
    pcd.points = od.utility.Vector3dVector(vtx)
    z = np.asarray(pcd.points)[:,2]
    z_norm = (z - z.min()) / (z.max() - z.min())
    colors = plt.get_cmap('jet')(z_norm)[:, :3]
    pcd.colors = od.utility.Vector3dVector(colors)
    od.visualization.draw_geometries([pcd])

    sensor.stop()
    sensor.close()

def get_dev(ctx):
    devices = ctx.query_devices()
    return devices[0]


def set_cfg(adv):
    with open('cfg.json') as f:
        json_str = f.read().strip()
    adv.load_json(json_str)


if __name__ == '__main__':
    main()

    colors = plt.get_cmap('jet')(z_norm)[:, :3]
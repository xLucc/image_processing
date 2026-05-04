import time
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import open3d as od
import h5py
import debugpy



def main():
    # debugpy.listen(('0.0.0.0', 4865))
    # debugpy.wait_for_client()
    try:
        pipe, unit, intrinsics, align = setup_camera()
        depth_list , _ = accumulate_data(pipe, align)
        depth = time_image_filter(depth_list)
        depth = depth * unit

        # desk_mask = (depth > 0.27) & (depth < 0.285)
        # desk_z = depth[mask]

        # print(f"Mittelwert: {desk_z.mean()*1000:.2f}mm")
        # print(f"Std: {desk_z.std()*1000:.3f}mm")
        # print(f"Min: {desk_z.min()*1000:.2f}mm")
        # print(f"Max: {desk_z.max()*1000:.2f}mm")

        xyz = calc_3d(depth, intrinsics)

        pcd = od.geometry.PointCloud()
        pcd.points = od.utility.Vector3dVector(xyz)
        od.visualization.draw_geometries([pcd])
    finally:
        pipe.stop()





# Use bilateral filter to smoothen the image, but keeping the edges sharp.
def apply_bilateral(img, amount_of_neighbours=9, sigma_color=10, sigma_space=30):
    img = np.nan_to_num(img, nan=0.0).astype(np.float32)
    return cv.bilateralFilter(img, amount_of_neighbours, sigma_color, sigma_space)


# Calculate the median for each pixel over a series of images.
def time_image_filter(img_list):

    if not isinstance(img_list, (list, np.ndarray)):
        raise ValueError
    
    img_list = img_list.tolist() if isinstance(img_list, np.ndarray) else img_list

    images = np.stack(img_list, axis=-1).astype(np.float32)
    images = np.where(images==0, np.nan, images)
    
    return np.nanmedian(images, axis=2)
    

def get_device(ctx):
    devices = ctx.query_devices()
    return devices[0]

def set_cfg(adv):
    with open('high_density.json') as f:
        json_str = f.read().strip()
    adv.load_json(json_str)

def get_frames(pipe, align):

    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    return depth_frame, color_frame
    

def get_data_from_frame(frame):
    return np.asarray(frame.get_data()).astype(np.float32)

def accumulate_data(pipe, align):
    depth_list = []
    color_list = []

    for i in range(10):

        depth_frame, color_frame = get_frames(pipe, align)
        depth_list.append(apply_bilateral(get_data_from_frame(depth_frame)))
        color_list.append(apply_bilateral(get_data_from_frame(color_frame)))

    return depth_list, color_list

def get_data(pipe):
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    depth = depth_frame.get_data()
    return np.asarray(depth)


def calc_3d(depth, intrinsics):

    h,w = depth.shape
    mask = (depth > 0.1) & (depth < 0.3)

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth.flatten()
    X = (u - intrinsics.ppx) * depth / intrinsics.fx 
    Y = (v - intrinsics.ppy) * depth / intrinsics.fy
    X = X.flatten()
    Y = Y.flatten()

    xyz = np.column_stack([X,Y,Z])
    return xyz[mask.flatten()]


def setup_camera():
    ctx = rs.context()
    pipe = rs.pipeline(ctx)
    dev = get_device(ctx)
    file = h5py.File('intrinsic.hdf5', 'r')
    advanced_mode = rs.rs400_advanced_mode(dev)

    # Start the advanced mode.
    while not advanced_mode.is_enabled():
        advanced_mode.toggle_advanced_mode(True)
        time.sleep(3.0)
        dev = get_device(ctx)
        advanced_mode = rs.rs400_advanced_mode(dev)

    set_cfg(advanced_mode)


    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    depth_table = advanced_mode.get_depth_table()
    depth_table.depthUnits = 500
    advanced_mode.set_depth_table(depth_table)
    time.sleep(0.5)

    profile = pipe.start(config)

    for _ in range(15):
        pipe.wait_for_frames()

    # color_d_set = file['intrinsic']
    depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

    unit = profile.get_device().first_depth_sensor().get_depth_scale()

    return pipe, unit, depth_intrinsics, align




if __name__ == '__main__':
    main()
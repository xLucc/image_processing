import math
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

def max_z(focal_length, baseline, disparity_shift):
    return (focal_length * baseline) / disparity_shift

def focal_length(x_res, HFOV):
    return 0.5 * (x_res / math.tan(math.radians(HFOV) / 2))

def min_z(focal_length, baseline, disparity_shift):
    return (focal_length * baseline) / (disparity_shift + 126)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
selection = pipeline.start(config)
depth_stream =  selection.get_stream(rs.stream.depth)
intrinsic = rs.video_stream_profile(depth_stream).get_intrinsics()
hfov = math.degrees(2* math.atan(intrinsic.width / (2* intrinsic.fx)))

arr = np.arange(1, 129)
max_z_arr = max_z(focal_length(1280, hfov), 55, arr)
min_z_arr = min_z(focal_length(1280, hfov), 55, arr)

idx = np.argmin(np.abs(min_z_arr - 190))
max_distance = max_z_arr[0]
min_distance = min_z_arr[0]
print(f'Idx: {idx}, Max: {max_distance}, Min: {min_distance}')

# def calc_scalar_scale_factor(n):
#     return math.sqrt(math.pi / 2) / math.sqrt(n)

# def calc_scale_factor(n):
#     return np.sqrt(math.pi / 2) / np.sqrt(n)

# def frame_time(n): 
#     return n / 30


# arr = np.arange(1, 91)
# scale_norm = calc_scale_factor(arr) / calc_scale_factor(arr).max()
# time_norm = frame_time(arr) / frame_time(arr).max()

# cost = 0.5 * scale_norm + 0.5 * time_norm
# n_opt = arr[np.argmin(cost)]
# print(n_opt, calc_scalar_scale_factor(10), frame_time(10))
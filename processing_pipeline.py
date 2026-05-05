import time
import warnings
import h5py
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import open3d as od
import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt.space import Integer, Real

# Searchroom
space = [
    Integer(1, 500, name="guided_radius"),
    Real(1e-6, 1e-2, name="guided_eps", prior="log-uniform"),
    Real(0, 1, name="sharp_lambda"),
    Real(0.001, 0.02, name="resudial_clamp"),
]

STD_WEIGHT = 0.65
RATING_WEIGHT = 0.35
STD_BASELINE = 2.3
ZMIN = 0.19
ZMAX = 0.28


def setup_camera():
    """
    Return:
        pipe, unit, depth_intrinsics, align
    """

    def set_cfg(adv):
        with open("high_density.json") as f:
            json_str = f.read().strip()
        adv.load_json(json_str)

    def get_device(ctx):
        devices = ctx.query_devices()
        return devices[0]

    ctx = rs.context()
    pipe = rs.pipeline(ctx)
    dev = get_device(ctx)
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
    for _ in range(50):
        pipe.wait_for_frames()
    # color_d_set = file['intrinsic']
    depth_intrinsics = rs.video_stream_profile(
        profile.get_stream(rs.stream.depth)
    ).get_intrinsics()

    unit = profile.get_device().first_depth_sensor().get_depth_scale()

    return pipe, unit, depth_intrinsics, align


def aquire_data(pipe, align, n_frames=12):

    def get_frame():
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        d = np.asarray(aligned_frames.get_depth_frame().get_data())
        c = np.asarray(aligned_frames.get_color_frame().get_data())

        return d.astype(np.float32), c

    # def apply_bilateral(img):
    #     return cv.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space)

    depth_frames = []
    color_frames = []

    for _ in range(n_frames):
        depth_frame, color_frame = get_frame()
        depth_frames.append(depth_frame)
        color_frames.append(color_frame)
        time.sleep(0.015)

    return depth_frames, color_frames[-1]


def time_filter(img_list):

    if not isinstance(img_list, (np.ndarray, list)):
        raise TypeError

    img_list = img_list if isinstance(img_list, list) else img_list.tolist()

    images = np.stack(img_list, axis=-1).astype(np.float32)
    images = np.where(images == 0, np.nan, images)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(images, axis=2), nan=0).astype(np.float32)


def calc_3d(depth, color, intrinsics):
    h, w = depth.shape
    mask = (depth > ZMIN) & (depth < ZMAX)

    color = cv.cvtColor(color, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    X = ((u - intrinsics.ppx) * depth / intrinsics.fx).flatten()
    Y = ((v - intrinsics.ppy) * depth / intrinsics.fy).flatten()
    Z = depth.flatten()

    xyz = np.column_stack([X, Y, Z])
    return xyz[mask.flatten()], color.reshape(-1, 3)[mask.flatten()]


def get_user_input():

    cmd_map = {"b": 1.0, "w": -1.0, "e": 0.0}

    while True:
        cmd = input(
            "Please insert, if the result was [b]etter, [w]orst, or [e]qual, than the last one. \n"
        )

        if cmd in cmd_map:
            return cmd_map[cmd]

        print("Please insert b, e, or w.")


def show_pc(xyz, color, min=ZMIN, max=ZMAX, n_color=30):

    pcd = od.geometry.PointCloud()
    pcd.points = od.utility.Vector3dVector(xyz)
    pcd.colors = od.utility.Vector3dVector(color)
    od.visualization.draw_geometries([pcd])


def calc_std(depth, desk_mm=(258, 265)):
    lo, hi = desk_mm[0] / 1000, desk_mm[1] / 1000
    mask = (depth > lo) & (depth < hi)
    vals = depth[mask]
    if len(vals) < 100:
        return 999.0

    return float(vals.std() * 1000)


def combined_objective(std_mm, rating):
    std_score = (std_mm - STD_BASELINE) / STD_BASELINE
    rating_score = -rating
    return STD_WEIGHT * std_score + RATING_WEIGHT * rating_score


def convert_guide(color):
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=5)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=5)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6)
    return cv.GaussianBlur(edge_mag, (3, 3), 0)


def apply_filter(guide, target, radius, eps, unit, clamp, sharp_lambda):
    src = target * unit
    smooth = cv.ximgproc.guidedFilter(guide=guide, src=src, radius=radius, eps=eps)
    resudial = src - smooth
    resudial = np.clip(resudial, -clamp, clamp)
    return src + sharp_lambda * resudial * guide


def main():

    print("=== Bilateral Filter Bayesian Optimization ===")
    print(
        f"Searchspace: guided_radius∈[1, 1280], guided_eps∈[1e-6, 1e-1], lambda∈[0.5, 3.0], clamp∈[1, 8]"
    )
    print(f"Baseline Std: {STD_BASELINE}mm  (d=5, sc=50, ss=10)")
    print()

    pipe, unit, intrinsic, align = setup_camera()
    optimizer = Optimizer(space, base_estimator="GP", n_initial_points=5, acq_func="EI")

    best_score = float("inf")
    best_params = None
    history = []

    n_iter = int(input("How many iterations? [suggested: 15-25]\n").strip())
    print()

    try:
        for i in range(n_iter):
            suggestion = optimizer.ask()
            guided_radius, eps, sharp_lambda, clamp = suggestion

            print(f"── Iteration {i+1}/{n_iter} ──────────────────────")
            print(
                f"   lambda={sharp_lambda:.2f}, clamp={clamp * 1000:.1f}mm, radius={int(guided_radius)}, eps={eps:.2e}"
            )

            depth_raw, color_raw = aquire_data(pipe, align)
            depth = time_filter(depth_raw)
            color = convert_guide(color_raw)
            depth_smooth = apply_filter(
                guide=color,
                target=depth,
                radius=guided_radius,
                eps=eps,
                unit=unit,
                clamp=clamp,
                sharp_lambda=sharp_lambda,
            )
            std_mm = calc_std(depth_smooth)

            print(f"   Std: {std_mm:.3f}mm")

            pcd, pcc = calc_3d(depth_smooth, color_raw, intrinsic)
            show_pc(pcd, pcc)

            rating = get_user_input()
            score = combined_objective(std_mm, rating)

            optimizer.tell(suggestion, score)
            history.append(
                (int(guided_radius), eps, sharp_lambda, clamp, std_mm, rating, score)
            )

            if score < best_score:
                best_score = score
                best_params = (guided_radius, eps, sharp_lambda, clamp)
                print("New optimum!")

            print()
    finally:
        pipe.stop()

    file = h5py.File("bayesian.hdf5", "w")
    file.create_dataset("history", np.asarray(history))

    print("=== Result ===")
    print(f"  guided_radius  = {best_params[0]}")
    print(f"  guided_eps     = {best_params[1]:.2e}")
    print(f"  sharp_lambda   = {best_params[2]:.2f}")
    print(f"  clamp = {best_params[3]*1000:.1f}mm")
    print(f"  Score: {best_score:.4f}")
    print()
    print("── History ───────────────────────────────────────────────────────────")
    print(
        f"{'#':>3}  {'radius':>6}  {'eps':>8}  {'lambda':>6}  {'clamp':>7}  {'Std':>7}  {'Rating':>9}  {'Score':>7}"
    )
    for j, (r, eps, lam, clamp, std, rat, sc) in enumerate(history):
        rat_str = {1.0: "better", 0.0: "equal", -1.0: "worse"}[rat]
        print(
            f"{j+1:>3}  {r:>6}  {eps:>8.2e}  {lam:>6.2f}  "
            f"{clamp*1000:>5.1f}mm  {std:>6.3f}mm  {rat_str:>9}  {sc:>7.4f}"
        )


if __name__ == "__main__":
    main()

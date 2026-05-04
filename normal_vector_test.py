import time
import warnings
import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pyrealsense2 as rs


LOWER_GREEN = np.array([35, 80, 30])   # Sättigung von 50 auf 80
UPPER_GREEN = np.array([85, 255, 180]) # Value-Obergrenze begrenzen




def aquire_data(pipe, align, n_frames=12):

    def get_frame():
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        d = np.asarray(aligned_frames.get_depth_frame().get_data())
        c = np.asarray(aligned_frames.get_color_frame().get_data())
        
        return d.astype(np.float32), c
    
    depth_frames = []
    color_frames = []

    for _ in range(n_frames):
        depth_frame, color_frame = get_frame()
        depth_frames.append(depth_frame)
        color_frames.append(color_frame)
        time.sleep(0.015)
    
    return depth_frames, color_frames[-1]




def setup_camera():

    '''
    Return:
        pipe, unit, depth_intrinsics, align
    '''
    def set_cfg(adv):
        with open('high_density.json') as f:
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
    depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

    unit = profile.get_device().first_depth_sensor().get_depth_scale()

    return pipe, unit, depth_intrinsics, align



def time_filter(img_list):

    if not isinstance(img_list, (np.ndarray, list)):
        raise TypeError
    
    img_list = img_list if isinstance(img_list, list) else img_list.tolist()

    images = np.stack(img_list, axis=-1).astype(np.float32)
    images = np.where(images==0, np.nan, images)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nan_to_num(np.nanmedian(images, axis=2), nan=0).astype(np.float32)
    
def segment_by_edges(rgb_bgr, min_area=3000):  # war 500
    gray = cv.cvtColor(rgb_bgr, cv.COLOR_BGR2GRAY)
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blurred = cv.GaussianBlur(gray, (7, 7), 0)  # stärker glätten
    edges = cv.Canny(blurred, threshold1=40, threshold2=100)  # war 20/60
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    masks = []
    for cnt in contours:
        if cv.contourArea(cnt) < min_area:
            continue
        mask = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [cnt], -1, 255, cv.FILLED)
        masks.append((mask, cnt))
    
    return edges, masks


def mask_to_pointcloud(mask, depth_m, intrinsics):
    ys, xs = np.where(mask > 0)
    zs = depth_m[ys, xs]
 
    # Ungültige Tiefenwerte ausfiltern
    valid = (zs > 0.05) & (zs < 5.0)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
 
    # Rückprojektion in 3D
    X = (xs - intrinsics.ppx) * zs / intrinsics.fx
    Y = (ys - intrinsics.ppy) * zs / intrinsics.fy
    Z = zs
 
    points = np.stack([X, Y, Z], axis=1)
    return points


def estimate_normal(points):
    if len(points) < 3:
        raise ValueError("Zu wenige Punkte für PCA.")
 
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
 
    normal = Vt[-1]  # kleinster Singulärwert
 
    # Vorzeichen: Normalenvektor zur Kamera zeigen lassen
    camera_origin = np.array([0.0, 0.0, 0.0])
    if normal @ (centroid - camera_origin) > 0:
        normal = -normal
 
    fit_residual = S[-1] / len(points)  # mittlere Abweichung von der Ebene
    return normal, centroid, fit_residual


def visualize_mask(rgb_bgr, mask_filled, mask_raw):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Segmentierung", fontsize=13, fontweight="bold")
 
    axes[0].imshow(cv.cvtColor(rgb_bgr, cv.COLOR_BGR2RGB))
    axes[0].set_title("RGB")
    axes[0].axis("off")
 
    axes[1].imshow(mask_raw, cmap="Greens")
    axes[1].set_title("HSV-Maske (roh)")
    axes[1].axis("off")
 
    axes[2].imshow(mask_filled, cmap="Greens")
    axes[2].set_title("Maske (gefüllt)")
    axes[2].axis("off")
 
    plt.tight_layout()
    plt.show()



def visualize_pointcloud(points, normal, centroid):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.2, 0.8, 0.3])
 
    # Normalenpfeil
    arrow_end = centroid + normal * 0.1
    arrow = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([centroid, arrow_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    arrow.paint_uniform_color([1.0, 0.0, 0.0])
 
    # Koordinatenrahmen
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=centroid)
 
    o3d.visualization.draw_geometries(
        [pcd, arrow, frame],
        window_name="Punktwolke + Normalenvektor",
        width=900, height=600
    )


def mask_to_pointcloud(mask, depth_m, intrinsics):
    ys, xs = np.where(mask > 0)
    zs = depth_m[ys, xs]
 
    # Ungültige Tiefenwerte ausfiltern
    valid = (zs > 0.05) & (zs < 5.0)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
 
    # Rückprojektion in 3D
    X = (xs - intrinsics.ppx) * zs / intrinsics.fx
    Y = (ys - intrinsics.ppy) * zs / intrinsics.fy
    Z = zs
 
    points = np.stack([X, Y, Z], axis=1)
    return points


def main():

    pipeline, unit, intrinsic, align = setup_camera()

    depth_frames, color = aquire_data(pipeline, align)
    depth = time_filter(depth_frames)
    depth_m = depth * unit

    # Kantenerkennung + Segmentierung
    edges, object_masks = segment_by_edges(color, min_area=500)

    # Debug-Visualisierung
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(cv.cvtColor(color, cv.COLOR_BGR2RGB))
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny Kanten")
    axes[1].axis("off")

    combined_mask = np.zeros(color.shape[:2], dtype=np.uint8)
    for mask, _ in object_masks:
        combined_mask = cv.bitwise_or(combined_mask, mask)
    axes[2].imshow(combined_mask, cmap="gray")
    axes[2].set_title(f"Objekte ({len(object_masks)} gefunden)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    # Pro Objekt: Punktwolke + Normalenvektor
    for i, (mask, cnt) in enumerate(object_masks):
        print(f"\n── Objekt {i} ────────────────────────────")
        points = mask_to_pointcloud(mask, depth_m, intrinsic)
        print(f"  3D-Punkte: {len(points)}")
        if len(points) < 10:
            print("  ✗ Zu wenige 3D-Punkte, übersprungen.")
            continue

        normal, centroid, residual = estimate_normal(points)
        print(f"  Normalenvektor : [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]")
        print(f"  Schwerpunkt    : [{centroid[0]:+.4f}, {centroid[1]:+.4f}, {centroid[2]:+.4f}]")
        print(f"  Fit-Residuum   : {residual:.6f} m")

        z_axis = np.array([0.0, 0.0, 1.0])
        angle_deg = np.degrees(np.arccos(np.clip(abs(normal @ z_axis), 0, 1)))
        print(f"  Winkel zu z    : {angle_deg:.2f}°")

        visualize_pointcloud(points, normal, centroid)


if __name__ == "__main__":
    main()
 

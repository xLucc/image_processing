import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import copy


# ---------------------------------
# Parameter
# ---------------------------------
GUIDED_RADIUS = 500         # beeinflusst die Glaettung der Flaechen durch den Bilateral (ausgenommen der Kanten aus RGB)
GUIDED_EPS = 1e-5           # steuert den Einfluss des Guide-Bilds (je kleiner EPS, umso mehr wird Guide miteinbezogen)
SHARP_LAMBDA = 1          # Gewicht der Unsharp Map
RESIDUAL_CLAMP = 0.02       # [m] Clamp fuer maximal valide Aenderung eines Tiefenwerts (sollte eher klein gehalten werden sonst Ueberschwingen)


# ---------------------------------
# RealSense Setup
# ---------------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align = rs.align(rs.stream.color)

# Warm-up
for _ in range(10):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
frames = align.process(frames)

depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_raw = np.asarray(depth_frame.get_data())
color_img = np.asarray(color_frame.get_data())

pipeline.stop()


# ---------------------------------
# RGB Anzeige
# ---------------------------------
cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
cv2.imshow("RGB Image", color_img)


# ---------------------------------
# Depth -> Meter
# ---------------------------------
depth_m = depth_raw.astype(np.float32) * depth_scale

# ---------------------------------
# Guided / Joint Bilateral Filter
# ---------------------------------
# ---------------------------------
# RGB -> Edge-Guide für Guided Filter
# ---------------------------------

# RGB -> Graustufen
gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
cv2.namedWindow("RGB as gray Image", cv2.WINDOW_NORMAL)
cv2.imshow("RGB as gray Image", gray)

# Sobel-Kanten
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

edge_mag = np.sqrt(gx * gx + gy * gy)

# Normalisieren auf [0,1]
edge_mag = edge_mag / (edge_mag.max() + 1e-6)

# Leicht glätten, damit der Guided Filter stabil bleibt
edge_guide = cv2.GaussianBlur(edge_mag, (3, 3), 0)

# Anzeige des Guides (SEHR wichtig zum Debuggen)
cv2.namedWindow("RGB Edge Guide", cv2.WINDOW_NORMAL)
cv2.imshow("RGB Edge Guide", edge_guide)

# ---------------------------------
# Guided Filter mit Edge-Guide
# ---------------------------------
if hasattr(cv2.ximgproc, "guidedFilter"):
    depth_smooth = cv2.ximgproc.guidedFilter(
        guide=edge_guide,
        src=depth_m,
        radius=GUIDED_RADIUS,
        eps=GUIDED_EPS
    )
else:
    print("Fallback: Joint Bilateral Filter nicht aktiv")

# ---------------------------------
# Unsharp Mask (metrisch korrekt)
# ---------------------------------
residual = depth_m - depth_smooth
residual = np.clip(residual, -RESIDUAL_CLAMP, RESIDUAL_CLAMP)
# depth_sharp = depth_m + SHARP_LAMBDA * residual
depth_sharp = depth_m + SHARP_LAMBDA * residual * edge_guide        # explizit nur die Kanten mitnehmen

# ---------------------------------
# Intrinsik für Punktwolke
# ---------------------------------
intrinsics = profile.get_stream(rs.stream.color) \
    .as_video_stream_profile().get_intrinsics()

fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

h, w = depth_m.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

# ---------------------------------
# Funktion: Depth -> Point Cloud
# ---------------------------------
def depth_to_pointcloud(depth, color):
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3)[:, ::-1] / 255.0  # BGR -> RGB

    valid = z.reshape(-1) > 0
    return points[valid], colors[valid]

# Original
pts_orig, cols_orig = depth_to_pointcloud(depth_m, color_img)

# Geschärft
pts_sharp, cols_sharp = depth_to_pointcloud(depth_sharp, color_img)

# ---------------------------------
# Open3D Punktwolken
# ---------------------------------
def show_pointcloud(pcd, title):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 1.0
    # opt.background_color = np.asarray([0.05, 0.05, 0.05])
    opt.background_color = np.asarray([1.0, 1.0, 1.0])  # weiß

    vis.run()
    vis.destroy_window()


# ---------------------------------
# 2D Depth Visualisierung
# ---------------------------------
def visualize_depth_2d(depth, title, vmin=None, vmax=None):
    depth_vis = depth.copy()

    # ungültige Tiefen maskieren
    depth_vis[depth_vis <= 0] = np.nan

    if vmin is None or vmax is None:
        vmin = np.nanpercentile(depth_vis, 5)
        vmax = np.nanpercentile(depth_vis, 95)

    depth_vis = np.clip(depth_vis, vmin, vmax)
    depth_vis = (depth_vis - vmin) / (vmax - vmin)

    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow(title, depth_vis)
    return vmin, vmax


# gleiche Skalierung für fairen Vergleich
vmin, vmax = visualize_depth_2d(depth_m, "Depth 2D - Original")
visualize_depth_2d(depth_sharp, "Depth 2D - Sharpened", vmin, vmax)

cv2.waitKey(1)


# -------------------------------
# Punktwolken erzeugen
# -------------------------------
pcd_orig = o3d.geometry.PointCloud()
pcd_orig.points = o3d.utility.Vector3dVector(pts_orig)
pcd_orig.colors = o3d.utility.Vector3dVector(cols_orig)

pcd_sharp = o3d.geometry.PointCloud()
pcd_sharp.points = o3d.utility.Vector3dVector(pts_sharp)
pcd_sharp.colors = o3d.utility.Vector3dVector(cols_sharp)

# --------------------------------
# Gleiche Transformation sicherstellen
# --------------------------------
pcd_sharp = copy.deepcopy(pcd_sharp)

# --------------------------------
# Anzeige
# --------------------------------
print("Zeige ORIGINAL Punktwolke (Fenster 1). Fenster schließen für nächste Ansicht.")
show_pointcloud(pcd_orig, "Original Depth Point Cloud")

print("Zeige GESCHÄRFTE Punktwolke (Fenster 2).")
show_pointcloud(pcd_sharp, "Sharpened Depth Point Cloud")
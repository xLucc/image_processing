import numpy as np
import h5py
from pathlib import Path
from camera import RealSenseCamera
from image_processing import ImageProcessing


def main():
    cam = RealSenseCamera(adv="setup_cfg.json")
    processor = ImageProcessing()
    mtx = load_intrinsics(Path("intrinsics.hdf5"))
    color = cam.get_rgb()[
        0
    ]  # Die funktion gibt eine liste von bilder zurück, deswegen das [0]
    depth = cam.get_depth(num_frames=11)

    # Auskommentieren, falls nicht gewollt
    depth = processor.median_filtering_over_time(depth)
    depth *= cam.unit  # Wird in meter konvertiert
    depth = processor.apply_guided_filter(depth, color)

    pcd = calc_pcd(depth, mtx)
    # Hier dann die speicher funktion.


def load_intrinsics(path: Path):
    file = h5py.File(path, "r")
    mtx = file["intrinsics"]["mtx"][:]
    return mtx


def calc_pcd(depth, mtx):
    h, w = depth.shape
    mask = (depth > 0.1) & (depth < 0.3)  # An die unit anpassen.

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth.flatten()
    X = (u - mtx[0, 2]) * depth / mtx[0, 0]
    Y = (v - mtx[1, 2]) * depth / mtx[1, 1]
    X = X.flatten()
    Y = Y.flatten()

    xyz = np.column_stack([X, Y, Z])
    return xyz[mask.flatten()]


if __name__ == "__main__":
    main()

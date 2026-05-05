import image_processing.calibrate_camera as cc
import argparse
import h5py
import threading
import rclpy
import sys
from pathlib import Path
from bin_picking.robot.node import RobotNode


def main(**kwargs):
    robot = RobotNode()
    node = threading.Thread(target=robot.spin, daemon=True)
    node.start()
    robot.wait_for_service()
    robot_poses = []
    event = threading.Event()
    try:
        path = cc.find_repo_root() / "data" / "calibration"
    except FileNotFoundError:
        path = Path(__file__).parent / "data" / "calibration"

    if not kwargs.get("calib_exists", False):
        quit_event = threading.Event()
        thread = threading.Thread(target=cc.main, kwargs=kwargs | {"event": event})
        thread.start()

        while not quit_event.is_set():
            event.wait()
            robot_poses.append(robot.robot_pose)
            print("Got robot pose")

        thread.join()
        print("Camera calibration complete.")

        file = h5py.File(path / "intrinsics.h5", "r")
        mtx = file["intrinsics"]["mtx"][:]
        dist = file["intrinsics"]["dist"][:]
        file.close()


if __name__ == "__main__":
    rclpy.init()
    remaining = rclpy.utilities.remove_ros_args(sys.argv[1:])

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--check_shape", type=int, nargs=2, default=(8, 6))
    parser.add_argument("--show", type=bool, default=True)
    parser.add_argument("--calib_exists", type=bool, default=False)

    main(**vars(parser.parse_args(remaining)))

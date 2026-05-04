import pyrealsense2 as rs
import cv2 as cv
import numpy as np



def get_cmd(max_tries=5):

    for i in range(max_tries):
        input_val = input('Press c for continue, or q for quit. \n')
        cmd = input_val.strip().lower()
        if cmd == 'c':
            return True
        elif cmd == 'q':
            return False
        else:
            print('Please insert valid command.')
    
    raise RuntimeError

def keep_img_or_del(max_tries=5):
    for i in range(max_tries):
        input_val = input('To keep the image please insert k, to discard press d. \n')
        cmd = input_val.strip().lower()
        if cmd == 'k':
            return True
        elif cmd == 'd':
            return False
        else:
            print('Please insert valid command.')
    
    raise RuntimeError





height = 1280
width = 720

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, height, width, rs.format.bgr8, 30)
profile = pipeline.start(config)
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsic = color_profile.get_intrinsics()
print(intrinsic)
pipeline.stop()


count = 0

input('Press enter to start.')

try: 

    while True:

        while True:
            frame = pipeline.wait_for_frames().get_color_frame()

            if not frame:
                continue
            
            img = np.asarray(frame.get_data())
            cv.imshow('image', img)

            key = cv.waitKey(1) & 0xFF

            if key == ord('s'):
                break
            elif key == ord('q'):
                raise KeyboardInterrupt
            

        if keep_img_or_del():
            count += 1

            cv.imwrite(f'Image_{count}_resolution_{height}x{width}.png', np.asarray(frame.get_data()))
        else:
            print('Discarded.')

        if not get_cmd():
            break

        print('Please reposition. \n')
        print(f'Image number {count} \n')

finally:
    pipeline.stop()
    cv.destroyAllWindows()
        
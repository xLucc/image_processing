import numpy as np
import cv2 as cv
import glob
import h5py


check_shape = (8, 6)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
objectPoints = []
imagePoints = []

object_points = np.zeros((1, check_shape[0] * check_shape[1], 3), np.float32)
object_points[0, :, :2] = np.mgrid[0:check_shape[0], 0:check_shape[1]].T.reshape(-1,2)

images = glob.glob('./high_res/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (check_shape[0], check_shape[1]), None)

    if ret == True:
        objectPoints.append(object_points)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
        imagePoints.append(corners2)

        # cv.drawChessboardCorners(img, (check_shape[0], check_shape[1]), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None, flags=cv.CALIB_FIX_K3)

file = h5py.File('intrinsic.hdf5', 'w')
file.create_dataset('intrinsics/mtx', data=mtx)
file.create_dataset('intrinsics/dist', data=dist)
file.flush()
file.close()



# print(f'Reprojectionerror: {ret:.4f} px')
# print(f'dist: {dist}')
# print(mtx)

# mtx_real = np.array([[927.813, 0.0, 656.09], [0.0, 927.765, 358.115], [0.0,0.0,1.0]])
# dist_real = np.zeros_like(dist)

# img_path = images[0]
# img = cv.imread(img_path)
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# x,y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('undist.png', dst)

import numpy as np
import cv2
import matplotlib.image as mpimg

def calibrate_camera(nx, ny, images):

    objpoints = [] #it's 3d point of objects
    imgpoints = [] #it's 2d points of calib images

    # Preparing object points (0,0,0) (0,0,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    #Create x,y coordinates using mgrid
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)

        # Convert BGR (because reading using opencv to grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Finding chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners found ret!=null then
        if ret == True:
            # Drawing corners
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def perspective_transform(img, is_inverse):
    img_size = (img.shape[1], img.shape[0])
    #source coordinates
    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1280, 720], [0, 720]])
    if is_inverse:
        M=cv2.getPerspectiveTransform(dst, src)
        view = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return view
    else:
        M = cv2.getPerspectiveTransform(src, dst)
        view = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return view



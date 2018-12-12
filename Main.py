import numpy as np
import cv2
import glob
import os
import pickle
import FindingLines as fl
import CheckingLines as chl
import CameraCalibration as cc
from Line import Line
import Curvature as cr
import matplotlib.pyplot as plt



# Count quantity of corners on chessboard
m_nx = 9
m_ny = 6
mtx = []
dist = []
#Getting images path
try:
    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle.get("mtx")
    dist = dist_pickle.get("dist")
except (OSError, IOError) as e:

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    m_images_path = os.path.join(PROJECT_ROOT, 'camera_cal', 'calibration*.jpg')
    # Choose images folder
    m_images = glob.glob(m_images_path)
    mtx, dist = cc.calibrate_camera(m_nx, m_ny, m_images)
    dist_pickle= {"mtx": mtx, "dist": dist}
    pickle.dump(dist_pickle, open("wide_dist_pickle.p", "wb"))
img_test = cv2.imread('calibration1.jpg')


cap = cv2.VideoCapture('project_video.mp4')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, size)

right_line = Line()
left_line = Line()
count=0
while cap.isOpened():
    ret, road_image = cap.read()
    if ret == True:
        count+=1
        road_image = cv2.cvtColor(road_image, cv2.COLOR_BGR2RGB)
        undist_road_img = cv2.undistort(road_image, mtx, dist, None, mtx)
        bird_view = cc.perspective_transform(undist_road_img, False)
        binary_road_img = fl.getting_binary_output(bird_view)


        left_line.detected,right_line.detected,left_line.allx,left_line.ally, right_line.allx, right_line.ally = chl.calculate(binary_road_img, left_line, right_line)

        if (left_line.detected):
            left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
            left_x_int, left_top = left_line.get_top_and_bottom(left_fit)

            #Averaging top and bottom
            left_line.x_int.append(left_x_int)
            left_line.top.append(left_top)
            left_x_int = np.mean(left_line.x_int)
            left_top = np.mean(left_line.top)
            left_line.lastx_int = left_x_int
            left_line.last_top = left_top

            #Add averaging top and bottom to avoid losing line

            left_x = np.append(left_line.allx, left_x_int)
            left_y = np.append(left_line.ally, 720)
            left_x = np.append(left_x, left_top)
            left_y = np.append(left_y, 0)

            #sort values with added
            left_x, left_y = left_line.sort(left_x,left_y)
            left_line.allx = left_x
            left_line.ally = left_y

            # Recalculate polynomial
            left_fit = np.polyfit(left_y, left_x, 2)
            left_line.fit0.append(left_fit[0])
            left_line.fit1.append(left_fit[1])
            left_line.fit2.append(left_fit[2])

            left_line.curvature.append(cr.get_curvature(left_line.allx, left_line.ally))



        if (right_line.detected):
            right_fit = np.polyfit(right_line.ally, right_line.allx, 2)
            right_x_int, right_top = right_line.get_top_and_bottom(right_fit)

            # Averaging top and bottom
            right_line.x_int.append(right_x_int)
            right_line.top.append(right_top)
            right_x_int = np.mean(right_line.x_int)
            right_top = np.mean(right_line.top)
            right_line.lastx_int = right_x_int
            right_line.last_top = right_top

            # Add averaging top and bottom to avoid losing line

            right_x = np.append(right_line.allx, right_x_int)
            right_y = np.append(right_line.ally, 720)
            right_x = np.append(right_x, right_top)
            right_y = np.append(right_y, 0)

            # sort values with added
            right_x, right_y = right_line.sort(right_x, right_y)
            right_line.allx = right_x
            right_line.ally = right_y

            # Recalculate polynomial
            right_fit = np.polyfit(right_y, right_x, 2)
            right_line.fit0.append(right_fit[0])
            right_line.fit1.append(right_fit[1])
            right_line.fit2.append(right_fit[2])

            right_line.curvature.append(cr.get_curvature(right_line.allx, right_line.ally))



        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_road_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, binary_road_img.shape[0] - 1, binary_road_img.shape[0])
        left_fitx = np.mean(left_line.fit0) * ploty ** 2 + np.mean(left_line.fit1) * ploty + np.mean(left_line.fit2)
        right_fitx = np.mean(right_line.fit0) * ploty ** 2 + np.mean(right_line.fit1) * ploty + np.mean(right_line.fit2)
        # Recast the x and y points into usable format for cv2.fillPoly()
        print("LEFT LINE")
        print(left_line.fit0)
        print(left_line.fit1)
        print(left_line.fit2)
        print("---------------------------------")

        print("RIGHT LINE")
        print(right_line.fit0)
        print(right_line.fit1)
        print(right_line.fit2)
        print("---------------------------------")
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
 #       plt.imshow(binary_road_img)
#        plt.show()
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cc.perspective_transform(color_warp, True)

        # Combine the result with the original image
        result = cv2.addWeighted(undist_road_img, 1, newwarp, 0.3, 0)
        result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        curvature=(np.mean(left_line.curvature))
        car_position=(cr.get_car_position(np.mean(left_line.fit0),np.mean(left_line.fit1),np.mean(left_line.fit2),np.mean(right_line.fit0),np.mean(right_line.fit1),np.mean(right_line.fit2),ploty))
        cv2.putText(result,car_position,(120,40),fontFace = 16, fontScale = 1, color=(255,255,255))
        cv2.putText(result, "Curve: "+str(curvature), (120, 80), fontFace=16, fontScale=1, color=(255, 255, 255))


        # write the flipped frame
        out.write(result)
        cv2.imshow('lines', result)
        if cv2.waitKey(25) & 0xFF == ord('q'):

            break

cap.release()
out.release()
cv2.destroyAllWindows()

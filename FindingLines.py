import cv2
import numpy as np
import matplotlib.pyplot as plt


def select_yellow(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([0,80,200])
    upper = np.array([40,255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    return mask

def select_white(image):
    lower = np.array([20,0,200])
    upper = np.array([255,80,255])
    mask = cv2.inRange(image, lower, upper)

    return mask

def getting_binary_output(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]


    # Threshold color channel
    l_thresh_min = 215
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1


    yellow = select_yellow(img)
    white = select_white(img)
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(l_binary)
    combined_binary[(l_binary == 1)| (b_binary ==1)| (yellow >= 1) | (white >= 1)] = 1

    return combined_binary



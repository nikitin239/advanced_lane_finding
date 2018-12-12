## Report

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Images/Calibration_2_pic_find_corners.png "FindCorners"
[image2]: ./Images/original_undistorted_img.png "Original-undistorted"
[image3]: ./Images/UndistortedRoad.png "Undistorted road"
[image4]: ./Images/bird_view_straight_lines.png "BirdView"

[image5]: ./Images/Sobelx_only.png "Sobelx_only"
[image6]: ./Images/GrayScaled.png "GrayScaled"
[image7]: ./Images/Lightness_only_in_HLS.png "Lightness only"
[image8]: ./Images/Only_HUE_channel.png "HUE only"
[image9]: ./Images/Only_saturation.png "Saturation only"
[image10]: ./Images/Original_to_Binary_(sobelx+saturation_thres).png "Saturation only"
[image11]: ./Images/original_unwrapped.png "Unwrapped"
[image12]: ./Images/binary_straight_lines.png "BirdView Binary"
[image13]: ./Images/only_s_binary.png "Only Saturation Binary"
[image14]: ./Images/s_binary+sobelx.png "s_binary+sobel"
[image15]: ./Images/Histogram.png "Histogram"
[image16]: ./Images/Finding_the_lines_polinom.png "Windows"
[image17]: ./Images/Finding_lines.png "lines"
[image18]: ./Images/train_s_binary.png "Train saturation"
[image19]: ./Images/train_sx_binary.png "Train SobelX"
[image20]: ./Images/Curvative_and_car_position.png "Curvative and car position"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---


### Camera Calibration
This process is implemented in camera_calibration function. You have to pass nx, ny parameters (corners) to it and array of calibration images. 
At first I prepare object points 

```python
# Preparing object points (0,0,0) (0,0,0)
    objp = np.zeros((nx*ny, 3), np.float32)
```
Then for each image i find chessboard corners and append founded cirners to imgpoints and objpoints
using cv2.calibrateCamera i've got calibration matrix and distortion coefficients which I save to pickle variable. 

```python
def camera_calibration(nx, ny, images):

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
```

![alt text][image1]

### Undistortion
After camera calibration I used undistortion method using MTX, DST Parameters, which I've got from previous step: 
```python
undist_road_img = cv2.undistort(image, mtx, dist, None, mtx)
```
For chessboard it looks like:
![alt text][image2]

And for road image:
![alt text][image3]



### Bird-View
After image undistortion we need to transform image using perspective. This functionality is implemented in perspective_transform function. 
Whe choose src and dst matrix in image pixels depended on image zone which we decided to be lane-lines zone and which we need to transform to bird view

Ex for corners chessboard:
![alt text][image11]
```python
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
```
If is_inverse parameter is false- we do perspective transform to bird view, if it is true - we calculate inverse matrix and return inverse transformed picture. 
There are one example: 

![alt text][image4]

### Finding Lines Methods
As we see there are some difficulties to detect yellow line for ex. from grayscaled picture using thresholds.
![alt text][image6]

At first I've tried to detect lines only using sobelx method to find lines from gradient picture:

![alt text][image5]

 we could extract more data from picture using another color scheme and combine methods. 

For ex. HLS-color scheme returned me this results by different channels:
#### Lightness only
![alt text][image7]
#### HUE only
![alt text][image8]
#### Saturation only
![alt text][image9]
#### SobelX+Saturation
![alt text][image10]


For my solution I've used white and yellow colors thresholds+ L-channel thresholds method (LUV) + B-channel threshold (LAB) (finish solution) Implemetation:

```python

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

```
There are ex in BirdView mode:
#### BirdView Binary FINISH SOLUTION (white thres+yellow thres+lchannel+b channel)
![alt text][image12]







### Finding Lines Methods
After converting images to binary we follow methods for building polinomial model of lines.
1. We build histogram to find left and right lines on x.

```python
# Take a histogram of the bottom half of the image to get x-coordinates, where will be a lot of '1' values- means
    #  lane lines
    histogram = np.sum(binary_warped[:,:], axis=0)

    # Create an output image
    # calculate middle of histogram resolution
    midpoint = np.int(histogram.shape[0]/2)
    # get max value of left side- means left line coord
    leftx_base = np.argmax(histogram[:midpoint])
    # get max value of right side - means right line coord
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
![alt text][image15]

Then splitting picture to 9 pieces we find left and right lines x,y coordinates on each window:
```python
# choose windows numbers
    nwindows = 9
    # Calculate each window height
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
```

And then find polynomial models:
```python
    # Fit a second order polynomial to each
 
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
```


### Calculate curvative
At first we need to define calculations to transform pixels to meters
```python
   ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
```

And then we could calculate curvative:
```python
def get_curvature(x, y):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

    curverad = round(((1 + (2*fit_cr[0]*np.max(y)*ym_per_pix + fit_cr[1])**2)**1.5)\
                                 /np.absolute(2*fit_cr[0]),1)
    return curverad

```

And Car's posiiton
```python
def get_car_position(left_fit0, left_fit1,left_fit2, right_fit0,right_fit1,right_fit2, ploty):
    #calculate car's line position:
    left_position = left_fit0 * np.max(ploty) ** 2 + left_fit1 * np.max(ploty) + left_fit2
    right_position = right_fit0 * np.max(ploty) ** 2 + right_fit1 * np.max(ploty) + right_fit2
    car_offset = (1280/2)-((left_position+right_position)/2)
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    meter_offset = round((abs(car_offset) * xm_per_pix),1)
    if (car_offset<0):
        output = 'car is: '+str(meter_offset)+' meter left of center'
    else:
        output="car is: "+str(meter_offset)+" meter right of center"
    return (output)
```
![alt text][image20]
### Video Processing
After getting all parameters I use my algorithm to video file: 
```python
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
```

To store the Line model i used class Line:
```python
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)



        # Count the number of frames
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.current_fit = None
        self.count = 0
        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=5)
        self.top = deque(maxlen=5)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None
        self.curvature = deque(maxlen=4)
    def is_detected(self):
        if self.detected:
            return True
        else:
            return False



    def sort(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals


    def get_top_and_bottom(self, polynomial):
        bottom = polynomial[0] * 720 ** 2 + polynomial[1] * 720 + polynomial[2]
        top = polynomial[0] * 0 ** 2 + polynomial[1] * 0 + polynomial[2]
        return bottom, top

```

For getting better mask results I've used averaging lines parameters by 5 measurements
This returned me better results, and it helped me avoid strange lines masks out of carriageway marking
The video is on the project folder

#### Proposals
I think it's possible to achieve better results after playing with different color schemes following sobel operator for different channels, also there will be a lot of problems at night because my solution now gets a lot of information from color thresholds(yellow,white)

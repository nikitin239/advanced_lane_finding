import numpy as np
def get_curvature(x, y):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

    curverad = round(((1 + (2*fit_cr[0]*np.max(y)*ym_per_pix + fit_cr[1])**2)**1.5)\
                                 /np.absolute(2*fit_cr[0]),1)
    return curverad
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
# Define a class to receive the characteristics of each line detection
from collections import deque

import numpy as np
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



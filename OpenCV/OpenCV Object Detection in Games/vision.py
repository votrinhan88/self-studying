import cv2
import numpy as np
from hsvfilter import HsvFilter

class Vision:
    # Trackbar window
    TRACKBAR_WINDOW = 'Trackbars'

    # property
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    def __init__(self, needle_img_path, method = cv2.TM_CCOEFF_NORMED):
        self.needle_img = cv2.imread(needle_img_path)

        # Save needle image dimension
        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        self.method = method

    def find(self, haystack_img, threshold = 0.8, max_results = 10):
        # Find 'Needle in a Haystack'
        match_result = cv2.matchTemplate(haystack_img, self.needle_img, self.method)
        # Filter with threshold
        locations = np.where(match_result >= threshold)

        # zip into tuples
        # array[start:stop:step]: slice to access list elements in reverse order
        # *: unpack a list (remove outer dimension into two list)
        # zip(): pack elements from two lists into pairs
        # list(): pack all pairs into a list
        locations = list(zip(*locations[::-1]))
        if not locations:
            return np.array([], dtype = np.int32).reshape(0, 4)

        # Group overlapping results
        rectangles = []
        for location in locations:
            rect = [int(location[0]), int(location[1]), self.needle_w, self.needle_h]
            rectangles.append(rect)
            # Mark twice so rectangles are not misdeleted
            rectangles.append(rect)
        rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold = 1, eps = 0.2)

        if len(rectangles) > max_results:
            print('Too many results, raise the thresholds.')
            rectangles = rectangles[0: max_results]
        return rectangles

    def get_center(self, rectangles):
        points = []
        for (x, y, w, h) in rectangles:
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            points.append((center_x, center_y))
        return points
    
    # Draw rectangle over results
    def draw_rectangles(self, haystack_img, rectangles):
        for (x, y, w, h) in rectangles:
            cv2.rectangle(haystack_img,
                        (x, y), (x + w, y + h),
                        color = (0, 255, 0), thickness = 1)
        return haystack_img

    # Draw crosshairs over results
    def draw_crosshairs(self, haystack_img, points):
        for (center_x, center_y) in points:
            cv2.drawMarker(haystack_img, (center_x, center_y),
                           markerType = cv2.MARKER_CROSS, markerSize = 10, color = (0, 0, 255))
        return haystack_img

    def init_control_gui(self):
        cv2.namedWindow(self.TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

        # A pseudo-callback (not used, getTrackbarPos() is used instead)
        def nothing(position):
            pass

        # Create tracks for bracketing
        # OpenCV scale for HSV: H:[0, 179], S:[0, 255], V:[0, 255]
        cv2.createTrackbar('h_min', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv2.createTrackbar('s_min', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv2.createTrackbar('v_min', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv2.createTrackbar('h_max', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv2.createTrackbar('s_max', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv2.createTrackbar('v_max', self.TRACKBAR_WINDOW, 0, 255, nothing)
        # Trackbars for increasing/decreasing S and V
        cv2.createTrackbar('s_add', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv2.createTrackbar('s_sub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv2.createTrackbar('v_add', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv2.createTrackbar('v_sub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        # Set default values
        cv2.setTrackbarPos('h_max', self.TRACKBAR_WINDOW, 179)
        cv2.setTrackbarPos('s_max', self.TRACKBAR_WINDOW, 255)
        cv2.setTrackbarPos('v_max', self.TRACKBAR_WINDOW, 255)

    def getHsvFilter(self):
        trackbars = {}
        for trackbar in ['h_min', 's_min', 'v_min', 'h_max', 's_max', 'v_max',
                         's_add', 's_sub', 'v_add', 'v_sub']:
            trackbars[trackbar] = cv2.getTrackbarPos(trackbar, self.TRACKBAR_WINDOW)
        
        hsvFilter = HsvFilter(**trackbars)
        return hsvFilter
    
    def applyHsvFilter(self, image, hsvFilter = None):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # If default filter is given, use current values in trackbar
        if not hsvFilter:
            hsvFilter = self.getHsvFilter()

        # Shift S and V
        h, s, v = cv2.split(image_hsv)
        s = self.shiftChannel(s, hsvFilter.s_add)
        s = self.shiftChannel(s, -hsvFilter.s_sub)
        v = self.shiftChannel(v, hsvFilter.s_add)
        v = self.shiftChannel(v, -hsvFilter.v_sub)
        image_hsv = cv2.merge([h, s, v])
        
        # Set min, max HSV values
        lower = np.array([hsvFilter.h_min, hsvFilter.s_min, hsvFilter.v_min])
        upper = np.array([hsvFilter.h_max, hsvFilter.s_max, hsvFilter.v_max])
        # Apply thresholds
        mask = cv2.inRange(image_hsv, lower, upper)
        result = cv2.bitwise_and(image_hsv, image_hsv, mask = mask)

        image = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        return image

    def shiftChannel(self, channel, amount):
        if amount > 0:
            lim = 255 - amount
            channel[channel >= lim] = 255
            channel[channel < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            channel[channel <= lim] = 0
            channel[channel < lim] -= amount
        return channel





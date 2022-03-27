import cv2
import numpy as np
from hsvfilter import HsvFilter
import re
from enemy import Enemy
# OCR
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class Vision:
    # Trackbar window
    TRACKBAR_WINDOW = 'Trackbars'

    # property
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    def __init__(self, enemy: Enemy, method = cv2.TM_CCOEFF_NORMED):
        self.enemy = enemy
        self.needle_img = cv2.cvtColor(cv2.imread(self.enemy.image_path), cv2.COLOR_BGR2GRAY)
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
    @classmethod
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
    
    @classmethod
    def applyHsvFilter(cls, image, hsvFilter = None, mode = 'in'):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # If default filter is given, use current values in trackbar
        if not hsvFilter:
            hsvFilter = cls.getHsvFilter()

        # Shift S and V
        h, s, v = cv2.split(image_hsv)
        s = cls.shiftChannel(cls, s, hsvFilter.s_add)
        s = cls.shiftChannel(cls, s, -hsvFilter.s_sub)
        v = cls.shiftChannel(cls, v, hsvFilter.s_add)
        v = cls.shiftChannel(cls, v, -hsvFilter.v_sub)
        image_hsv = cv2.merge([h, s, v])
        
        # Set min, max HSV values
        lower = np.array([hsvFilter.h_min, hsvFilter.s_min, hsvFilter.v_min])
        upper = np.array([hsvFilter.h_max, hsvFilter.s_max, hsvFilter.v_max])
        # Apply thresholds
        if mode == 'in':
            mask = cv2.inRange(image_hsv, lower, upper)
        elif mode == 'out':
            mask = cv2.bitwise_not(cv2.inRange(image_hsv, lower, upper))
        
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
    
    def ocr(self, haystack_img, rectangles):
        if rectangles.any():
            (x, y, w, h) = rectangles[np.random.randint(0, high=len(rectangles))]
            # Text image
            needle_image = haystack_img[y:y+h, x:x+w]
            cv2.rectangle(needle_image, (self.enemy.x_left, self.enemy.y_up), (self.enemy.x_right, self.enemy.y_bottom),
                        color = (0, 255, 0), thickness = 1)
            cv2.imshow(f'Needle ({self.enemy.name})', needle_image)

            text_image = needle_image[self.enemy.y_up:self.enemy.y_bottom, self.enemy.x_left:self.enemy.x_right]
            # text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
            _, text_image = cv2.threshold(text_image, 63, 255, cv2.THRESH_BINARY)
            cv2.imshow(f'Text ({self.enemy.name})', text_image)

            text = pytesseract.image_to_string(text_image, lang = 'eng', config = '-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM --psm 8 --oem 3')
            text_fixed = re.sub('[^A-Z]', '', text)
            return text_fixed

class VisionBonus():
    def __init__(self, enemy: Enemy):
        self.enemy = enemy
        
    def ocr(self, haystack_img):
        (x, y, w, h) = (self.enemy.x_left, self.enemy.x_right, self.enemy.y_up, self.enemy.y_bottom)
        text_image = haystack_img[y:y+h, x:x+w]
        _, text_image = cv2.threshold(text_image, 191, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow(f'Text ({self.enemy.name})', text_image)
        if 0 in text_image:
            text = pytesseract.image_to_data(text_image, lang = 'eng', config = '-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM --psm 7 --oem 3')
            text_fixed = re.sub('[^A-Z]', '', text)
            return text_fixed


import cv2
import numpy as np
from time import time

from windowcapture import WindowCapture


def findClickPosition(haystack_img_path, needle_img_path, threshold = 0.8, debug_mode = None):
    # Find 'Needle in a Haystack'
    haystack_img = cv2.imread(haystack_img_path)        #, flags = cv2.IMREAD_GRAYSCALE)
    needle_img = cv2.imread(needle_img_path)            #, flags = cv2.IMREAD_GRAYSCALE)
    match_result = cv2.matchTemplate(haystack_img, needle_img, cv2.TM_CCOEFF_NORMED)
    # Filter with threshold
    locations = np.where(match_result >= threshold)

    # zip into tuples
    # array[start:stop:step]: slice to access list elements in reverse order
    # *: unpack a list (remove outer dimension into two list)
    # zip(): pack elements from two lists into pairs
    # list(): pack all pairs into a list
    locations = list(zip(*locations[::-1]))

    # Group overlapping results
    rectangles = []
    for location in locations:
        rect = [int(location[0]), int(location[1]), needle_img.shape[1], needle_img.shape[0]]
        rectangles.append(rect)
        # Mark twice so rectangles are not misdeleted
        rectangles.append(rect)
    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold = 1, eps = 0.5)

    # Draw rectangle over results
    if rectangles.any():
        print(f'{len(rectangles)} needles found.')
        
        points = []
        for (x, y, w, h) in rectangles:
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                cv2.rectangle(haystack_img,
                              (x, y), (x + w, y + h),
                              color = (0, 255, 0), thickness = 1)
            elif debug_mode == 'points':                                        
                cv2.drawMarker(haystack_img, (center_x, center_y),
                               markerType = cv2.MARKER_CROSS, markerSize = 10, color = (0, 0, 255))
    if debug_mode != None:
        cv2.imshow('Match result', haystack_img)
        cv2.waitKey(0)
    
    return points

# points = findClickPosition(haystack_img_path = 'src/screen.jpg',
#                            needle_img_path = 'src/shark_basic.jpg',
#                            debug_mode = 'points')
# print(points)

def getFPS():
    loop_time = time()
    while (True):
        screenshot = wincap.capture()
        print(f'FPS: {1/(time() - loop_time):.2f}')
        loop_time = time()
        cv2.imshow('Screenshot', screenshot)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

wincap = WindowCapture('Typer Shark Deluxe 1.02')
# wincap.list_window_names()
getFPS()
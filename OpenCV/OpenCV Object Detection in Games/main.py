import cv2
from time import time
from cv2 import rectangle
from windowcapture import WindowCapture
from vision import Vision

# WindowCapture.list_window_names()
needle_shark = Vision(needle_img_path = 'src/shark_basic.jpg')
needle_shark.init_control_gui()

wincap = WindowCapture('Typer Shark Deluxe 1.02')

loop_time = time()
while (True):
    screenshot = wincap.capture()
    print(f'FPS: {1/(time() - loop_time):.2f}')
    loop_time = time()
    screenshot = needle_shark.applyHsvFilter(screenshot)
    rectangles = needle_shark.find(screenshot, threshold = 0.6)
    display = needle_shark.draw_rectangles(screenshot, rectangles)
    

    cv2.imshow('Result', display)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
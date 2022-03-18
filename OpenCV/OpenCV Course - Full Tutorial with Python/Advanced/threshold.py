import cv2
import numpy as np

image = cv2.imread('Resources/Photos/cats.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simple thresholding
threshold, thresh = cv2.threshold(image_gray,
                                  thresh = 127,
                                  maxval = 255,
                                  type = cv2.THRESH_BINARY)
threshold, thresh_inv = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)                                  
# Adaptive thresholding
thresh_adapt = cv2.adaptiveThreshold(image_gray,
                                        maxValue = 255,
                                        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType = cv2.THRESH_BINARY,
                                        blockSize = 11,
                                        C = 5)
# Display
display = np.vstack((np.hstack((image,
                                np.stack((thresh,)*3, axis = 2))),
                     np.hstack((np.stack((thresh_inv,)*3, axis = 2),
                                np.stack((thresh_adapt,)*3, axis = 2)))))
display = cv2.resize(display, dsize = None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Original, Simple Thresholding, Simple Inversed Thresholding, Adaptive Thresholding', display)
cv2.waitKey(0)
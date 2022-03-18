import cv2
import numpy as np

image = cv2.imread('Resources/Photos/park.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Laplacian 
laplacian = cv2.Laplacian(image_gray,
                          ddepth = cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Sobel (is a stage within Canny, seldom use)
sobelx = cv2.Sobel(image_gray,
                   ddepth = cv2.CV_64F,
                   dx = 1,
                   dy = 0)
sobely = cv2.Sobel(image_gray,
                   ddepth = cv2.CV_64F,
                   dx = 0,
                   dy = 1)
sobel_combined = cv2.bitwise_or(sobelx, sobely)

# Canny (frequently used)
canny = cv2.Canny(image_gray,
                  threshold1 = 150,
                  threshold2 = 175)

# Display
display = np.hstack((laplacian, sobel_combined, canny))
display = cv2.resize(display, dsize = None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Original (Park)', image)
cv2.imshow('Laplacian, Sobel combined, Canny', display)

cv2.waitKey(0)
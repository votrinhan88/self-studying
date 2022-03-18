import cv2
import numpy as np

image = cv2.imread('Resources/Photos/cats.jpg')
# cv2.imshow('Cats', image)

# Averaging
average = cv2.blur(image, (5, 5))

# Gaussian blur (looks more natural)
gaussian = cv2.GaussianBlur(image, (5, 5), 0)

# Median blur (can preserve the edge, frequently used)
median = cv2.medianBlur(image, 5)

# Bilateral blur (can preserve the edge, frequently used)
# looks like a washed out painting
bilateral = cv2.bilateralFilter(image, 10, 35, 25)

row1 = np.hstack((average, gaussian))
row2 = np.hstack((median, bilateral))
display = np.vstack((row1, row2))
cv2.imshow('Average, Gaussian, Median, Bilateral Blur', display)

cv2.waitKey(0)
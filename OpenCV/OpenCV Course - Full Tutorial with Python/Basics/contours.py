import cv2
import numpy as np

image = cv2.imread('Resources/Photos/cats.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Draw contours from Threshold
ret, threshold = cv2.threshold(image_gray, 125, 255, cv2.THRESH_BINARY)
cv2.imshow('Cats threshold', threshold)

blank = np.zeros(image.shape, dtype = 'uint8')
contours, hierarchies = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f'Found {len(contours)} contours.')
cv2.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv2.imshow('Contours drawn (threshold)', blank)

# Draw contours from Canny (is more recommended)
image_blur = cv2.GaussianBlur(image_gray, (5, 5), cv2.BORDER_DEFAULT)
image_canny = cv2.Canny(image_blur, 125, 175)

blank_2 = np.zeros(image.shape, dtype = 'uint8')
contours, hierarchies = cv2.findContours(image_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f'Found {len(contours)} contours.')
# Approximately redraw image_canny
cv2.drawContours(blank_2, contours, -1, (0, 0, 255), 1)
cv2.imshow('Contours drawn (Canny)', blank_2)


cv2.waitKey(0)
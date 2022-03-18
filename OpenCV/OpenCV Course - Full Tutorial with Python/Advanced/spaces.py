import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('Resources/Photos/park.jpg')
# Convert to different color spaces
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
display = np.hstack((image,
                     np.stack((image_gray,)*3, axis = 2),
                     image_hsv,
                     image_lab))
display = cv2.resize(display, dsize = None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Original, Grayscale, HSV, LAB', display)

# If used with other libraries (eg Matplotlib), change back to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)

# Inverse from different color spaces back to BGR
hsv_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
lab_bgr = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
display = np.hstack((hsv_bgr, lab_bgr))
display = cv2.resize(display, dsize = None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('HSV -> BGR, LAB -> BGR', display)

plt.show()
cv2.waitKey(0)
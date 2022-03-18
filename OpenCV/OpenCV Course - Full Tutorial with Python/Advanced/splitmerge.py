import cv2
import numpy as np

image = cv2.imread('Resources/Photos/park.jpg')
blank = np.zeros(image.shape[:2], dtype = 'uint8')

# Split color channels
b, g, r = cv2.split(image)
print(f'Original shape: {image.shape}')
print(f'Shapes of B, G, R channels: {b.shape}, {g.shape}, {r.shape}')

blue = cv2.merge([b, blank, blank])
green = cv2.merge([blank, g, blank])
red = cv2.merge([blank, blank, r])

display = np.hstack((blue, green, red))
display = cv2.resize(display, dsize = None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Blue, Green, Red', display)

# Merge color channels
bgr = cv2.merge([b, g, r])
cv2.imshow('Merged back to BGR', bgr)

cv2.waitKey(0)
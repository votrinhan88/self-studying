import cv2
import numpy as np

image = cv2.imread('Resources/Photos/park.jpg')

cv2.imshow('Park', image)

# Translation
def translate(image, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, transMat, dimensions)

image_translated = translate(image, x = 75, y = 50)
cv2.imshow('Park translated', image_translated)

# Rotation
def rotate(image, angle, pivot = None):
    (height, width) = image.shape[:2]
    if pivot is None:
        pivot = (width//2, height//2)
    rotMat = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    dimensions = (width, height)
    return cv2.warpAffine(image, rotMat, dimensions)

image_rotated = rotate(image, 30, None)
# Rotating clips pixels outside of frame
image_rotated_rotated = rotate(image_rotated, 60, None)
cv2.imshow('Park rotated twice', image_rotated_rotated)

# Resize
image_resized = cv2.resize(image, (500, 500), interpolation = cv2.INTER_CUBIC)
cv2.imshow('Park resized', image_resized)

# Flip: 0 for horizontal, 1 for vertical, -1 for both
image_flipped = cv2.flip(image, -1)
cv2.imshow('Park flipped', image_flipped)

# Crop
image_cropped = image[100:300, 200:400]
cv2.imshow('Park cropped', image_cropped)

cv2.waitKey(0)
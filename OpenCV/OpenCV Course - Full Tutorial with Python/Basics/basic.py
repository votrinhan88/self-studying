import cv2

image = cv2.imread('Resources/Photos/park.jpg')
# cv2.imshow('Park', image)

# Convert to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Park grayscale', image_gray)

# Blur
# increase blurness by increasing kernel size
image_blur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
# cv2.imshow('Park blurred', image_blur)

# Edge cascade
image_canny = cv2.Canny(image_blur, 125, 175)
cv2.imshow('Park blurred, cannied', image_canny)

# Dilate
image_dilated = cv2.dilate(image_canny, (5, 5), iterations = 3)
cv2.imshow('Park blurred, cannied, dilated', image_dilated)
# Erode: With same kernel and iterations, can approximately revert dilation
image_eroded = cv2.erode(image_dilated, (5, 5), iterations = 3)
cv2.imshow('Park blurred, cannied, dilated, eroded', image_eroded)

# Resize
# Downsize: use cv2.INTER_AREA
# Upsize: use cv2.INTER_LINEAR or cv2.INTER_CUBIC (higher quality, slower)
image_resized = cv2.resize(image, (500, 500), interpolation = cv2.INTER_AREA)
cv2.imshow('Park resized', image_resized)

# Crop
image_cropped = image[100:300, 200:400]
cv2.imshow('Park cropped', image_cropped)

cv2.waitKey(0)
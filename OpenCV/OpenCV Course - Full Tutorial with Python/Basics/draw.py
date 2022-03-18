import cv2
import numpy as np

blank = np.zeros((500, 500, 3), dtype = 'uint8')
# cv2.imshow('Blank', blank)

# Paint to green using Numpy
blank[100:300, 200:425] = 0, 255, 0
# cv2.imshow('Green', blank)

# Draw a rectangle
cv2.rectangle(blank, (50, 50), (150, 200), (0, 0, 255), 5)
# thickness = -1 for solid color
cv2.rectangle(blank, (100, 125), (125, 75), (255, 0, 0), -1)

# Draw a circle
cv2.circle(blank, (275, 275), 40, (127, 127, 0), 5)
cv2.circle(blank, (275, 275), 25, (127, 0, 127), -1)

# Draw a line
cv2.line(blank, (50, 400), (150, 300), (0, 127, 127), 8)

# Draw text
cv2.putText(blank, 'draw.py putText', (200, 450),
            fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 1,
            color = (191, 191, 0), thickness = 2)

cv2.imshow('Drawn', blank)
cv2.waitKey(0)
# image = cv2.imread('Resources/Photos/cat.jpg')
import cv2
import numpy as np
    
blank = np.zeros((400, 400), dtype = 'uint8')

rectangle = cv2.rectangle(blank.copy(), (50, 50), (350, 350), 255, -1)
circle = cv2.circle(blank.copy(), (200, 200), 175, 255, -1)

# Bitwise AND, OR, NOT, XOR
bitwise_and = cv2.bitwise_and(rectangle, circle)
bitwise_or = cv2.bitwise_or(rectangle, circle)
bitwise_not = cv2.bitwise_not(rectangle)
bitwise_xor = cv2.bitwise_xor(rectangle, circle)
display = np.vstack((np.hstack((bitwise_and, bitwise_or)), np.hstack((bitwise_not, bitwise_xor))))
cv2.imshow('Bitwise AND, OR, NOT, XOR', display)

cv2.waitKey(0)
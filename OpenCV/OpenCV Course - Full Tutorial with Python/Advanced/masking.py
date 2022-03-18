from tkinter import image_names
import cv2
import numpy as np

cats = cv2.imread('Resources/Photos/cats.jpg')
blank_cats = np.zeros(cats.shape[:2], dtype = 'uint8')
mask_circle = cv2.circle(blank_cats, (cats.shape[1]//2, cats.shape[0]//2), 175, 255, -1)
masked_cats = cv2.bitwise_and(cats, cats, mask = mask_circle)
cv2.imshow('Masked cats', masked_cats)

cats_2 = cv2.imread('Resources/Photos/cats 2.jpg')
blank_cats_2 = np.zeros(cats_2.shape[:2], dtype = 'uint8')
# Combining two masks or more: must make a copy of blank background
# If not, cv2 will draw permanently onto the background --> bitwise AND
mask_rectangle = cv2.rectangle(blank_cats_2.copy(),
                               (50, 25),
                               (500, 300),
                               255, -1)
mask_circle_2 = cv2.circle(blank_cats_2.copy(), (275, 300), 300, 255, -1)      
mask_combined = cv2.bitwise_and(mask_rectangle, mask_circle_2)                      
masked_cats_2 = cv2.bitwise_and(cats_2, cats_2, mask = mask_combined)
cv2.imshow('Masked cats 2', masked_cats_2)

cv2.waitKey(0)
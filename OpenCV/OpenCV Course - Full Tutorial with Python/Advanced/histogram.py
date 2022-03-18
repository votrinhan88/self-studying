import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Resources/Photos/cats.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask
blank = np.zeros(image.shape[:2], dtype = 'uint8')
circle = cv2.circle(blank.copy(), (image.shape[1] // 2, image.shape[0] // 2), 150, 255, -1)
masked_gray = cv2.bitwise_and(image_gray, image_gray, mask = circle)
masked_color = cv2.bitwise_and(image, image, mask = circle)

display = np.hstack((np.stack((masked_gray,)*3, axis = 2), masked_color))
cv2.imshow('Masked grayscale, Masked color', display)

# Grayscale histogram
gray_hist = cv2.calcHist(images = [image_gray],
                         channels = [0],
                         mask = circle,
                         histSize = [256],
                         ranges = [0, 256])

# Color histogram
colors = ('blue', 'green', 'red')
color_hist = {}
for i, color in enumerate(colors):
    color_hist[color] = cv2.calcHist(images = [image],
                                     channels = [i],
                                     mask = circle,
                                     histSize = [256],
                                     ranges = [0, 256])

# Plotting
figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
# Fig 1: Grayscale histogram
axes[0].set_title('Grayscale histogram (masked)')
axes[0].set_xlabel('Bins')
axes[0].set_ylabel('# of labels')
axes[0].set_xlim([0, 256])
axes[0].plot(gray_hist)
# Fig 2: Color histogram
axes[1].set_title('Color histogram (masked)')
axes[1].set_xlabel('Bins')
axes[1].set_ylabel('# of labels')
axes[1].set_xlim([0, 256])
for color in colors:
    axes[1].plot(color_hist[color], color = color)

plt.show()
cv2.waitKey(0)




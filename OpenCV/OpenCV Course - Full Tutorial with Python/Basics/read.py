import cv2

# Reading images
image = cv2.imread('Resources/Photos/cat.jpg')
# image = cv2.imread('Resources/Photos/cat_large.jpg') # Will not compress itself to fit the screen
cv2.imshow('Cat', image)
cv2.waitKey(0)

# Reading videos
capture = cv2.VideoCapture('Resources/Videos/dog.mp4')
while True:
    isTrue, frame = capture.read()
    cv2.imshow('Video', frame)

    # Close window when press D
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv2.destroyAllWindows()


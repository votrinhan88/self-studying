import cv2

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)

def changeResolution(width, height):
    # Only work with live videos
    capture.set(3, width)
    capture.set(4, height)

# Reading images
image = cv2.imread('Resources/Photos/cat_large.jpg')
image_resized = rescaleFrame(frame = image, scale = 0.25)
cv2.imshow('Cat large', image)
cv2.imshow('Cat large resized', image_resized)
cv2.waitKey(0)

# Reading videos
capture = cv2.VideoCapture('Resources/Videos/dog.mp4')
while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame = frame, scale = 0.5)
    cv2.imshow('Dog', frame)
    cv2.imshow('Dog resized', frame_resized)

    # Close window when press D
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv2.destroyAllWindows()
import cv2

haarcascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

def face_detect(image_name):
    image = cv2.imread(f'Resources/Photos/{image_name}.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rect = haarcascade.detectMultiScale(image_gray,
                                              scaleFactor = 1.1,
                                              minNeighbors = 3)
    print(f'Picture: {image_name}.jpg . Number of faces found: {len(faces_rect)}')
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(f'Detected faces in {image_name}.jpg', image)

for image_name in ('lady', 'group 1', 'group 2'):
    face_detect(image_name)

cv2.waitKey(0)
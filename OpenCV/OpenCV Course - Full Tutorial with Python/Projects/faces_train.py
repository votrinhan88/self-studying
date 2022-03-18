import os
import cv2
import numpy as np

haarcascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

DIR = r'C:\Users\openit\Desktop\opencv\Resources\Faces\train'
people = []
for i in os.listdir(DIR):
    people.append(i)

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            image_array = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            faces_rect = haarcascade.detectMultiScale(image_gray, scaleFactor = 1.1, minNeighbors = 4)
            for (x, y, w, h) in faces_rect:
                faces_roi = image_gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done -----------------')

features = np.array(features, dtype = 'object')
labels = np.array(labels)
print(f'Length of the features: {len(features)}')
print(f'Length of the labels: {len(labels)}')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('Scripts/Projects/face_trained.yml')
np.save('Scripts/Projects/features.npy', features)
np.save('Scripts/Projects/labels.npy', labels)


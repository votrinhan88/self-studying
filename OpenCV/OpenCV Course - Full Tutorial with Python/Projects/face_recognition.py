import os
import cv2
import numpy as np

haarcascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
DIR = r'C:\Users\openit\Desktop\opencv\Resources\Faces\train'
people = []
for i in os.listdir(DIR):
    people.append(i)

features = np.load('Scripts/Projects/features.npy', allow_pickle=True)
labels = np.load('Scripts/Projects/labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Scripts/Projects/face_trained.yml')

image = cv2.imread('Resources/Faces/val/ben_afflek/3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Person', image_gray)

# Detect the face in the image
faces_rect = haarcascade.detectMultiScale(image_gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = image_gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {label}, confidence = {confidence:.2f}%')
    cv2.putText(image, str(people[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Predicted', image)

cv2.waitKey(0)
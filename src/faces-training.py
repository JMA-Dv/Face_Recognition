import os
from PIL import Image
import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR,"images")

current_id = 0
label_ids = {}

y_labels = []
x_train = []

for root, dirs, files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-").lower()
            #print(path, label)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id = label_ids[label]
            #print(label_ids)

            pil_image = Image.open(path).convert("L")#grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(final_image,"uint8")#turning every image into numpy
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            #35:03
            for (x,y,w,h) in faces:
                roi= image_array[y:y+h , x:x + w]
                x_train.append(roi)
                y_labels.append(id)



#print(y_labels)
#print('name= ', x_train)

with open("labels.pickle", 'wb') as fl:
    pickle.dump(label_ids, fl)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")



import numpy as np
import os
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}

with open('labels.pickle', 'rb') as f:
    og_labels= pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}



cap = cv2.VideoCapture(0)
while(True):
    #capture frame by frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #region oof interest
    for (x, y, w, h) in faces:
       # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]#cuts the noise and focuses on the face
        roi_color = frame[y:y+h, x:x+w]

        #recognize deep learned model predict
        id,conf = recognizer.predict(roi_gray)
        if conf>=45:# and conf <=85:
            print(id)
            print(labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)




        img_item = "7.png"
        cv2.imwrite(img_item,roi_color)

        color = (255, 0, 200 )#
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h

        cv2.rectangle(frame, (x,y),(end_coord_x, end_coord_y),color,stroke)

    #Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()



import os
import cv2
import random

path_original = "../dataset_original"
path_destination = "../dataset"
files_sub = os.listdir(path_original)
print(files_sub)
face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_alt2.xml')
for i in range(len(files_sub)):
    files = os.listdir(path_original+"/"+files_sub[i])
    k = 1;
    for j in range(len(files)):
        k = k * 10
        try:
            img = cv2.imread(path_original+"/"+files_sub[i]+"/"+files[j])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            print(faces)
            for(x,y,w,h) in faces:
                roi_color = img[y:y+h, x:x+w]
                print(path_destination+"/"+files_sub[i]+"/"+files[j])
                cv2.imwrite(path_destination+"/"+files_sub[i]+"/"+files[j], roi_color)
                cv2.imwrite(path_destination+"/"+files_sub[i]+"/"+str(random.randint(0,500))+".jpg", roi_color)
                cv2.imwrite(path_destination+"/"+files_sub[i]+"/"+str(random.randint(0,500))+".jpg", roi_color)
                cv2.imwrite(path_destination+"/"+files_sub[i]+"/"+str(random.randint(0,500))+".jpg", roi_color)
                cv2.imwrite(path_destination+"/"+files_sub[i]+"/"+str(random.randint(0,500))+".jpg", roi_color)
        except Exception as error:
            print(error)
            # pass

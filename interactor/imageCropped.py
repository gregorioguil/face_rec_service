import os
import cv2

path_original = "../dataset_original"
path_destination = "../dataset"
files_sub = os.listdir(path_original)
face_cascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_alt2.xml')
for i in range(len(files_sub)):
    files = os.listdir(path_original+"/"+files_sub[i])
    for j in range(len(files)):
        try:
            img = cv2.imread(path_original+"/"+files_sub[i]+"/"+files[j])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for(x,y,w,h) in faces:
                roi_color = img[y:y+h, x:x+w]
                cv2.imwrite(path_destination+"/"+files_sub[i]+"/"+files[j], roi_color)
        except Exception as error:
            print(error)
            # pass

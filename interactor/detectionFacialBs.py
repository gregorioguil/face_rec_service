import os
import cv2
from werkzeug.utils import secure_filename

class DetectionFacialBs:
    def __init__(self, photos):
        self.photos = photos

    def detect(self):
        faceClassifier = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')

        self.photos.save(secure_filename(self.photos.filename))
        image = cv2.imread(self.photos.filename)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceClassifier.detectMultiScale(gray, 1.3, 4)

        for x, y, h, w in faces:
            roi_color = image[y:y+h, x:x+w]
            cv2.imwrite('new-image.jpg', roi_color)
        
        os.remove(self.photos.filename)
        if len(faces) > 0:
            return {
                'face_detected': True
            }

        return {
            'face_detected': False
        }        
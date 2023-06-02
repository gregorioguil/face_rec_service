import cv2
from werkzeug.utils import secure_filename

class DetectionFacialBs:
    def __init__(self, photos):
        self.photos = photos

    def detect(self):
        xml_cascade = 'haarcascade_frontalface_alt2.xml'

        print(cv2.__version__)
        faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + xml_cascade)
        print(self.photos.filename)

        self.photos.save(secure_filename(self.photos.filename))
        image = cv2.imread(self.photos.filename)
        cv2.imwrite('new-image.jpg', image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceClassifier.detectMultiScale(gray)
        print(faces)

        for x, y, h, w in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

        if len(faces) > 0:
            return {
                'face_detected': True
            }

        return {
            'face_detected': False
        }        
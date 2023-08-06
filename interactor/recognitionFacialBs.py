import os
import numpy as np
import tensorflow as tf
from interactor.detectionFacialBs import DetectionFacialBs

class RecognitionFacialBs:
    def __init__(self, photo, model):
        self.photo = photo
        self.model = model
    
    def recognize(self):
        try:
            img_height = 240
            img_width = 240
            detection = DetectionFacialBs(self.photo)
            resultDetection = detection.detect()
            
            if resultDetection['face_detected'] == False:
                return {
                    'face_detected': False
                }
            path="./dataset/"
            train_ds = tf.keras.utils.image_dataset_from_directory(
                path,
                validation_split=0.8,
                subset="training",
                seed=123,
                image_size=(img_height,img_width),
                batch_size=64)

            class_names = train_ds.class_names            

            img = tf.keras.utils.load_img(
                "new-image.jpg", target_size=(img_height, img_width)
            )
            
            img = tf.keras.utils.img_to_array(img)
            img = tf.expand_dims(img, 0)

            predictions = self.model.predict(img)
            score = tf.nn.softmax(predictions[0])

            name = class_names[np.argmax(score)]
            confiance = 100 * np.max(score)
            
            return {
                'name': name,
                'percent_confidence': confiance,
                'face_detected': True
            }
        except Exception as error:
            print("error ", error)
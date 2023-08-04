import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from interactor.detectionFacialBs import DetectionFacialBs

class RecognitionFacialBs:
    def __init__(self, photo):
        self.photo = photo
        #self.model = create_model()
    
    def recognize(self):
        print('Inicio')
        try:
            model = load_model("models/model.h5")
            img_height = 240
            img_width = 240
            print("Carregou model")
            detection = DetectionFacialBs(self.photo)
            resultDetection = detection.detect()
            print(resultDetection)
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
            
            img3 = tf.keras.utils.load_img(
                path+"Isabel/1.jpeg", target_size=(240, 240)
            )

            img_array = tf.keras.utils.img_to_array(img3)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "1This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

            img = tf.keras.utils.load_img(
                "new-image.jpg", target_size=(240, 240)
            )
            
            img = tf.keras.utils.img_to_array(img)
            img = tf.expand_dims(img, 0)

            # print(img)
            print("Predição")
            predictions = model.predict(img)

            # print(predictions)
            score = tf.nn.softmax(predictions[0])


            print(np.argmax(score))
            print(100 * np.max(score))
            print(class_names)
            name = class_names[np.argmax(score)]
            confiance = 100 * np.max(score)
            print(
                "0This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(name, confiance)
            )
            
            return {
                'name': name,
                'percent_confidence': confiance,
                'face_detected': True
            }
        except Exception as error:
            print("error ", error)
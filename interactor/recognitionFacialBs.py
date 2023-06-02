import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, utils, applications, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from interactor.detectionFacialBs import DetectionFacialBs

class RecognitionFacialBs:
    def __init__(self, photo):
        self.photo = photo
        #self.model = create_model()
    
    def recognize(self):
        print('Inicio')
        try:
            detection = DetectionFacialBs(self.photo)
            resultDetection = detection.detect()
            print(resultDetection)
            # model = Sequential()

            # pretrained_model = tf.keras.applications.EfficientNetB7(input_shape=(96,96,3), include_top=False, weights="imagenet")

            # for layer in pretrained_model.layers:
            #     layer.trainable=False

            # model.add(pretrained_model)
            # model.add(layers.GlobalAveragePooling2D())
            # model.add(layers.Dropout(0.3))

            # model.add(layers.Dense(1))


            img =  tf.keras.utils.load_img(
                "./new-image.jpg", target_size=(180, 180)
            )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (96,96))
            #img = img/255.0
            # img = tf.concat(img, axis=0)
            img = tf.keras.utils.img_to_array(img)
            img = tf.expand_dims(img, 0)
            # img = tf.image.resize(img, [180, 180])
            print('Leu mensagem')
            model = load_model("model.h5")
            print("Carregou model")
            model.summary()

            # print(img)
            predictions = model.predict(img)

            print(predictions)
            score = tf.nn.softmax(predictions[0])


            print(np.argmax(score))
            print(100 * np.max(score))
            class_names = os.listdir("./dataset")
            print(class_names)
            name = class_names[np.argmax(score)]
            confiance = 100 * np.max(score)
            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(name, confiance)
            )

            # model.load_weights(latest)

            return {
                'name': name,
                'percent_confidence': confiance,
                'face_detected': True
            }
        except Exception as error:
            print("error ", error)
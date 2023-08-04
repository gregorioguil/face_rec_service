from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow as tf
import numpy as np
import os


path="../dataset/"
img_height = 240
img_width = 240

train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.8,
  subset="training",
  seed=123,
  image_size=(img_height,img_width),
  batch_size=64)

class_names = train_ds.class_names

print(class_names)

model = load_model("../models/model.h5")


img = tf.keras.utils.load_img(
  path+"angelina_jolie/498.jpg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img3 = tf.keras.utils.load_img(
  path+"bernardo/6.jpeg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img3)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
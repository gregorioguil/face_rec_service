import numpy as np
import os

from keras import layers, callbacks, utils, applications, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow as tf

img_height = 240
img_width = 240

files=os.listdir("../dataset")
print(files)
print(tf.__version__)
image_array=[]
label_array=[]
path="../dataset/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height,img_width),
  batch_size=64)

class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=64)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


ckp_path="trained_model/model"

model_checkpoint= tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                     monitor="accuracy",
                                                     save_best_only=True,
                                                     save_weights_only=True)

reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                              monitor="accuracy",
                                              mode="auto",
                                              cooldown=0,
                                              patience=5,
                                              verbose=1,
                                              min_lr=1e-6)


epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.summary()

model.save("../models/model.h5")
print('Model saved.')

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

img1 = tf.keras.utils.load_img(
  path+"mohamed_ali/2.jpg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img1)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img2 = tf.keras.utils.load_img(
  path+"brad_pitt/1.jpg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img2)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img3 = tf.keras.utils.load_img(
  path+"Isabel/5.jpeg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img3)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img4 = tf.keras.utils.load_img(
  path+"Isabel/1.jpeg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img4)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

img5 = tf.keras.utils.load_img(
  path+"Isabel/2.jpeg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img5)
img_array = tf.expand_dims(img_array, 0)
  
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
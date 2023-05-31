import numpy as np
import pandas as pd
import os
import cv2

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import layers, callbacks, utils, applications, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
# from keras.models import Model, load_model
import tensorflow as tf
#import efficientnet.tfkeras as efc

img_height = 180
img_width = 180

files=os.listdir("./dataset")
print(files)
print(tf.__version__)
image_array=[]
label_array=[]
path="./dataset/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.8,
  subset="training",
  seed=123,
  image_size=(img_height,img_width),
  batch_size=64
)

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

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
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

# for i in range(len(files)):
#   file_sub=os.listdir(path+files[i])
#   for k in tqdm(range(len(file_sub))):
#     try:
#       img = cv2.imread(path+files[i]+"/"+file_sub[k])
#       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#       img = cv2.resize(img, (96,96))
#       image_array.append(img)
#       label_array.append(i)
#     except:
#       pass
    
# image_array = np.array(image_array)/255.0
# label_array = np.array(label_array)

# X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.15)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# sess = tf.compat.v1.Session(config=config)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# if physical_devices:
#   try:
#     tf.config.set_logical_device_configuration(
#         physical_devices[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     print(e)

# model = Sequential()
# pretrained_model = tf.keras.applications.EfficientNetB7(input_shape=(96,96,3), include_top=False, weights="imagenet")

# for layer in pretrained_model.layers:
#     layer.trainable=False
    
# model.add(pretrained_model)
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dropout(0.3))

# model.add(layers.Dense(1))

model.summary()

# model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

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

# Epoch=300
# Batch_Size=64
# history=model.fit(X_train, Y_train,
#                   validation_data=(X_test,Y_test),
#                   batch_size=Batch_Size,
#                   epochs=Epoch,
#                   callbacks=[model_checkpoint, reduce_lr])

epochs=300
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open("model.tflite", "wb") as f:
#   f.write(tflite_model)
  
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

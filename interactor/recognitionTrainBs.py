import numpy as np
import pandas as pd
import os
import cv2

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import layers, callbacks, utils, applications, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow as tf


files = os.listdir("../dataset")
image_array=[]
label_array=[]
path="../dataset/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.8,
  subset="training",
  seed=123,
  image_size=(96, 96),
  batch_size=32)

class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(96, 96),
  batch_size=32)

for i in range(len(files)):
    file_sub=os.listdir(path+files[i])
    for k in tqdm(range(len(file_sub))):
        try:
            img = cv2.imread(path+files[i]+"/"+file_sub[k])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96,96))
            image_array.append(img)
            label_array.append(i)
        except:
            pass

print(label_array)
image_array = np.array(image_array)/255.0
label_array = np.array(label_array)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

pretrained_model=tf.keras.applications.EfficientNetB7(input_shape=(96,96,3), 
                                                      include_top=False,
                                                     weights="imagenet")

model = Sequential([
  pretrained_model,
  layers.GlobalAveragePooling2D(),
  layers.Dropout(0.3),
  layers.Dense(num_classes)
])

model.summary()
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

ckp_path="trained_model/model"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                            monitor="accuracy",
                                            save_best_only=True,
                                            save_weights_only=True)

reduce_lr= tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                            monitor="accuracy",
                                            mode="auto",
                                            cooldown=0,
                                            patience=5,
                                            verbose=1,
                                            min_lr=0.5)

# Epoch=300
# Batch_Size=64
# history=model.fit(X_train, Y_train,
#                 validation_data=(X_test, Y_test),
#                 batch_size=Batch_Size,
#                 epochs=Epoch,
#                 callbacks=[model_checkpoint, reduce_lr])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[model_checkpoint, reduce_lr]
)


model.load_weights(ckp_path)
# scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save("model.h5")
print("Saved model to disk")

img = cv2.imread("../new-image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = tf.keras.utils.img_to_array(img)
img = tf.expand_dims(img, 0)
img = tf.image.resize(img, [96, 96])

predictions = model.predict(img)

print(predictions)

score = tf.nn.softmax(predictions[0])

print(class_names)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)




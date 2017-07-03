import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.applications.inception_v3 import InceptionV3

lines = []
with open('driving_log.csv', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
steering_correction = 0.4
for line in lines:
    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1]) #line[1][1:]
    image_right = cv2.imread(line[2])
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    steering_center = float(line[3])
    steering_left = steering_center + steering_correction
    steering_right = steering_center - steering_correction
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

augmented_images, augmented_measurements = [], []
#augmented_images = images
#augmented_measurements = measurements
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

#model.add(Convolution2D(32, 7, 7, subsample=(2,2), border_mode='same', activation="relu"))
#model.add(Convolution2D(24, 7, 7, subsample=(2,2), border_mode='same', activation="relu"))
model.add(Convolution2D(36, 7, 7, subsample=(2,2), border_mode='same', activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', activation="relu"))
model.add(Convolution2D(64, 5, 5, subsample=(2,2), border_mode='same', activation="relu"))
model.add(Convolution2D(80, 3, 3, subsample=(2,2), border_mode='same', activation="relu"))
model.add(Convolution2D(80, 3, 3, subsample=(2,2), border_mode='same', activation="relu"))
model.add(Flatten())
model.add(Dense(120, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')

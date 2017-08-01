import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.applications.inception_v3 import InceptionV3

lines = []
# A steering bias (positive or negative) for the left and right cameras
steering_correction = 0.4

#Read the driving (training) data collected from simulator
with open('driving_log.csv', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#Split the data test to have a training set and a validation set
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#Define a generator to better manage the large set of training data (images)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            #Using the left and right camera images to augment the training
            for line in batch_samples:
                image_center = cv2.imread(line[0])
                image_left = cv2.imread(line[1])  # line[1][1:]
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
            # augmented_images = images
            # augmented_measurements = measurements

            #Flipping the training data to balance the left-turn heavy track
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #yield sklearn.utils.shuffle(X_train, y_train)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()

#ch, row, col = 3, 80, 320  # Trimmed image format
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))

#Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))

#Trim image to only see section with road
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
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3*2, validation_data=validation_generator,
            nb_val_samples=len(validation_samples)*3*2, nb_epoch=2)
model.save('model.h5')

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
steering_correction = 0.1
for line in lines:

    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1])
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

model.add(Convolution2D(16, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(32, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


#model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
#model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
#model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
#model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Flatten())
#model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
##model.add(Dense(5))
#model.add(Dense(1))


#model.add(Convolution2D(6, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))

#model.add(Flatten(input_shape=(160, 320, 3)))
#model.add(Dense(100))
#model.add(Activation('relu'))
#model.add(Dense(60))
#model.add(Activation('relu'))
#model.add(Flatten())


model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

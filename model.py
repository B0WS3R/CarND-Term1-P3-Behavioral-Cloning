########################################################################################################################
# Udacity Self-Driving Car Nanodegree
# Term1-P3 Behavioral cloning
########################################################################################################################

import core
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda

########################################################################################################################
# main program
########################################################################################################################

# get generate data from the folders (using .csv)
data, labels = core.generate_data()

# Model hyper-parameters

batch = 64
epochs = 7
activation_type = 'relu'
dropout = .25

# Model architecture

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255) - .5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation_type))
model.add(Dropout(dropout))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation_type))
model.add(Dropout(dropout))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation_type))
model.add(Dropout(dropout))
model.add(Conv2D(64, (3, 3), activation=activation_type))
model.add(Dropout(dropout))
model.add(Conv2D(64, (3, 3), activation=activation_type))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(1164, activation=activation_type))
model.add(Dense(100, activation=activation_type))
model.add(Dense(50, activation=activation_type))
model.add(Dense(10, activation=activation_type))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(data, labels, batch_size=batch, epochs=epochs, validation_split=.2)

# save the trained model
model.save('model.h5')
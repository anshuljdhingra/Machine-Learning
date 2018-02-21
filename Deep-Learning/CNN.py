# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:01:44 2017

@author: adhingra
"""
# Convolution Neural Network

# importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initlaising the CNN
classifier = Sequential()

# adding convoluton layer
classifier.add(Convolution2D(32,(3,3), input_shape = (64,64,3), activation= 'relu'))

# adding max pooling layer
classifier.add(MaxPooling2D(pool_size= (2,2) ))

# adding a second convolutional and pooling layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# flatttening the pooled feature maps
classifier.add(Flatten())

# adding fully connected layer
classifier.add(Dense(output_dim = 128, activation= 'relu'))

# adding output layer
classifier.add(Dense(units= 1, activation= 'sigmoid'))

# compiling the CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# fitting the CNN model to the dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=20,
                        validation_data=test_set,
                        nb_val_samples=2000)

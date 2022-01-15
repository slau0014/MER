import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import torch
import torch.nn as nn
import deepfool
import model as mdl
from torchvision import transforms
import torchvision

K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 64, 64, 18

training_list = []

# training path
negativepath = '/content/drive/MyDrive/MeSelves/FYP_1/micro-expression/negative/'
positivepath = '/content/drive/MyDrive/MeSelves/FYP_1/micro-expression/positive/'
surprisepath = '/content/drive/MyDrive/MeSelves/FYP_1/micro-expression/surprise/'

print("Start processing training data")
# read frames from videos
# negative train dataset
directorylisting = os.listdir(negativepath)
for video in directorylisting:
    videopath = negativepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

# positive train dataset
directorylisting = os.listdir(positivepath)
for video in directorylisting:
    videopath = positivepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

# surprise train dataset
directorylisting = os.listdir(surprisepath)
for video in directorylisting:
    videopath = surprisepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

# pre-process of training data
training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)
traininglabels = numpy.zeros((trainingsamples, ), dtype = int)

traininglabels[0:65] = 0     # negative
traininglabels[65:112] = 1   # positive
traininglabels[112:153] = 2  # surprise

traininglabels = np_utils.to_categorical(traininglabels, 3)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))

for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

print("Finish processing training data")

# MicroExpSTCNN Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

model.summary()

filepath="/content/drive/MyDrive/MeSelves/FYP_1/micro-expression/weights_microexpstcnn/smic/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("Start training")
avg = 0.0
for i in range(1,6):
    print("Iteration: ", i)
    # Spliting the dataset into training and validation sets
    train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=i)

    # Training the model
    hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

    # Finding Confusion Matrix using pretrained weights
    predictions = model.predict(validation_images)
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(validation_labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    acc = (cfm[0][0]+cfm[1][1]+cfm[2][2])/32
    print(acc)
    avg += acc

print(avg/5)
print("Finish training")
print()

# Testing part
test_negativepath = '/content/drive/MyDrive/MeSelves/EVP_results/smic_label/negative/'
test_positivepath = '/content/drive/MyDrive/MeSelves/EVP_results/smic_label/positive/'
test_surprisepath = '/content/drive/MyDrive/MeSelves/EVP_results/smic_label/surprise/'

test_list = []

# process testing data
print("Start processing testing data")

directorylisting = os.listdir(test_negativepath)
for dir in directorylisting:
  for video in os.listdir(test_negativepath + dir):
    frames = []
    videopath = test_negativepath + dir + '/' + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(18)]
    for frame in framerange:
      image = loadedvideo.get_data(frame)
      imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
      grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
      frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    test_list.append(videoarray)

directorylisting = os.listdir(test_positivepath)
for video in directorylisting:
        frames = []
        videopath = test_positivepath + video
        loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
        framerange = [x for x in range(18)]
        for frame in framerange:
                image = loadedvideo.get_data(frame)
                imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        test_list.append(videoarray)

directorylisting = os.listdir(test_surprisepath)
for video in directorylisting:
        frames = []
        videopath = test_surprisepath + video
        loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
        framerange = [x for x in range(18)]
        for frame in framerange:
                image = loadedvideo.get_data(frame)
                imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
                grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        test_list.append(videoarray)

test_list = numpy.asarray(test_list)
testsamples = len(test_list)

testlabels = numpy.zeros((testsamples, ), dtype = int)

testlabels[0:119] = 0   # negative
testlabels[119:149] = 1  # positive
testlabels[149:173] = 2  # surprise

testlabels = np_utils.to_categorical(testlabels, 3)

test_data = [test_list, testlabels]
(testframes, testlabels) = (test_data[0], test_data[1])
test_set = numpy.zeros((testsamples, 1, image_rows, image_columns, image_depth))
for h in range(testsamples):
	test_set[h][0][:][:][:] = testframes[h,:,:,:]

test_set = test_set.astype('float32')
test_set -= numpy.mean(test_set)
test_set /= numpy.max(test_set)
print("Finish processing testing data")

predictions = model.predict(test_set)
predictions_labels = numpy.argmax(predictions, axis=1)
testlabels = numpy.argmax(testlabels, axis=1)
cfm = confusion_matrix(testlabels, predictions_labels)
print (cfm)

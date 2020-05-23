#!/usr/bin/env python
# coding: utf-8

from keras.datasets import mnist
from keras.utils import to_categorical
def load_dataset():
    (trainX,trainy),(testX,testy) = mnist.load_data()
    trainX = trainX.reshape(trainX.shape[0],28,28,1)
    testX = testX.reshape(testX.shape[0],28,28,1)
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    return trainX,trainy,testX,testy

def prep_pixels(train,test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    return train_norm,test_norm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam
def cnn_model():
    model = Sequential()
    model.add(Convolution2D(filters=32,kernel_size=(2,2),strides=(1,1),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

import numpy as np
from sklearn.model_selection import KFold
def evaluate_model(dataX, dataY, n_folds=2):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model = cnn_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
        score = np.array(scores)
        historie = np.array(histories)
    return score, historie

from numpy import mean
from numpy import std
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

import pandas as pd
def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)
    df1 = pd.DataFrame(scores,columns=['accuracy'])
    print(df1)
    df1.to_csv('result.csv')
    summarize_performance(scores)

run_test_harness()


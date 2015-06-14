#coding=utf8
from __future__ import absolute_import
import theano.tensor as T
import theano
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import time
import sys
import os
from sklearn import datasets
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.utils import np_utils


if __name__ == '__main__':
    n_samples = 5000
    n_features = 50
    n_informative = 20
    n_classes = 2
    x,y = datasets.make_classification(n_samples = n_samples,n_informative=n_informative,n_features = n_features,n_classes = n_classes)
    X_train = x[:-100]
    y_train = y[:-100]
    X_test = x[-100:]
    y_test = y[-100:]
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    model = Sequential()
    model.add(Dense(n_features, 2*n_informative, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2*n_informative, n_informative, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(n_informative, n_classes, init='uniform'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    start_time = time.clock()
    model.fit(X_train, y_train, nb_epoch=200, batch_size=50)
    end_time = time.clock()
    score = model.evaluate(X_test, y_test, batch_size=50, verbose=1, show_accuracy=True)
    print 'ran for %.1fs:' % (end_time - start_time)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
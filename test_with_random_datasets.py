import theano.tensor as T
import theano
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
from util import *
import time
import sys
import os
from sklearn import datasets




if __name__ == '__main__':
    n_samples = 30000
    n_features = 100
    n_informative = 30
    n_classes = 5
    x,y = datasets.make_classification(n_samples = n_samples,n_informative=n_informative,n_features = n_features,n_classes = n_classes)
    learning_rate = 0.13
    n_epochs = 200
    batch_size = 500  
    train_part,validation_part,test_part = (n_samples * 7 / 8,n_samples * 7 / 8,n_samples * 15 / 16)
    #mlp(x,y,n_features,50,n_classes,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,True)
   # own_write_regression_early_stopping(x.astype(theano.config.floatX),y.astype('int32'),n_features,n_classes,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,True)
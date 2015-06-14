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
    
    learning_rate = 0.13
    n_epochs = 200
    batch_size = 20  # size of the minibatch
    f = gzip.open('mini_mnist.pkl.gz', 'rb')
    x = cPickle.load(f)
    y = cPickle.load(f)
    n_feature,n_class=(784,10)
    f.close()
    train_part,validation_part,test_part = (1000,1000,1100)
 #   own_write_regression_early_stopping(x.astype(theano.config.floatX),y.astype('int32'),n_feature,n_class,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,True)
    rng = numpy.random.RandomState(1234)
    mlp(x,y,n_feature,150,n_class,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,True)

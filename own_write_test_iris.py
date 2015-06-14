import theano.tensor as T
import theano
import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from util import *


if __name__ == '__main__':
    
    learning_rate = 0.13
    n_epochs = 100
    batch_size = 15
    print 1
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    n_feature,n_class=(4,3)
    train_part,validation_part,test_part = (150,100,125)
    own_write_regression(x,y,n_feature,n_class,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,True)
    


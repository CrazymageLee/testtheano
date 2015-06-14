#coding=utf8
import theano.tensor as T
import theano
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
from logistic_sgd import LogisticRegression, load_data
import time
import codecs
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class logi_re(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(numpy.random.rand(n_in, n_out), name='W')  # @UndefinedVariable
        self.b = theano.shared(numpy.zeros(n_out,), name='b')  # @UndefinedVariable
        self.prob_y = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.pred_y = T.argmax(self.prob_y, axis=1)
        self.params = [self.W, self.b]

    def nll(self, y):
        return  -T.mean(T.log(self.prob_y)[T.arange(y.shape[0]),y])
    def error(self, y):
        return T.mean(T.neq(self.pred_y,y))

class hidden_layer(object):
    def __init__(self, rng, input, n_in, n_out, activation = T.tanh):
        '''
        W = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            '''
        W = rng.uniform(low=-numpy.sqrt(6./(n_in+n_out)),high=numpy.sqrt(6./(n_in+n_out)),size=(n_in, n_out))  # @UndefinedVariable
        if activation == T.nnet.sigmoid:
            W = W * 4
        self.W = theano.shared(W, name='W')  # @UndefinedVariable
        self.b = theano.shared(numpy.zeros(n_out,), name='b')  # @UndefinedVariable
        self.output = T.dot(input,self.W)+self.b
        if activation is not None:
            self.output = activation(self.output)
        self.params=[self.W,self.b]

class convpoollayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        fin = numpy.prod(filter_shape[1:])
        fout = filter_shape[0]*numpy.prod(filter_shape[2:])/numpy.prod(poolsize)
        w_bound = numpy.sqrt(6./(fin+fout))
        self.input = input
        self.W = theano.shared(numpy.array(rng.uniform(low=-w_bound,high=w_bound,size=filter_shape),dtype=theano.config.floatX), name='W')  # @UndefinedVariable
        self.b = theano.shared(numpy.zeros((filter_shape[0]),dtype=theano.config.floatX), name='W',borrow=True)   #@UndefinedVariable  
        conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=filter_shape,image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize,ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]
     


def mlp(x,y,n_feature,n_hidden,n_class,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,with_regularization):
    '''
    datasets = load_data('mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    '''
    train_set_x = theano.shared(numpy.asarray(x[:train_part],dtype=theano.config.floatX),name='train_set_x')  # @UndefinedVariable
    train_set_y = T.cast(theano.shared(y[:train_part],name='train_set_y'),'int32')
    valid_set_x = theano.shared(numpy.asarray(x[validation_part:test_part],dtype=theano.config.floatX),name='valid_set_x')  # @UndefinedVariable
    valid_set_y = T.cast(theano.shared(y[validation_part:test_part],name='valid_set_y'),'int32')
    test_set_x = theano.shared(numpy.asarray(x[test_part:],dtype=theano.config.floatX),name='test_set_x')  # @UndefinedVariable
    test_set_y = T.cast(theano.shared(y[test_part:],name='test_set_y'),'int32')
    # accessing the third minibatch of the training set
    # Load the dataset
    xx = T.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    yy = T.ivector('y')
    index = T.lscalar()
    
    rng = numpy.random.RandomState(1234)
    hl = hidden_layer(rng, xx, n_feature, n_hidden, activation = T.tanh)
    logi = logi_re(hl.output,n_hidden,n_class)
    lamb = 0.01
    if with_regularization:
        cost = logi.nll(yy)+lamb*T.sqrt(T.sum(logi.W**2))+lamb*T.sqrt(T.sum(hl.W**2))
    else:
        cost = logi.nll(yy)
    
    #为什么这里是加不能是[hl.params,logi.params],而在内部就是[]    
    params = hl.params+logi.params
        
    grad_params = [T.grad(cost, param) for param in params]
    
    updates = [(param,param - learning_rate*grad) for param,grad in zip(params,grad_params)]
    
    vali_error = theano.function(inputs = [], outputs=logi.error(yy),givens={xx: valid_set_x,yy: valid_set_y})
    
    test_error = theano.function(inputs = [], outputs=logi.error(yy),givens={xx: test_set_x,yy: test_set_y})
    
    test_result = theano.function(inputs = [], outputs=logi.pred_y, givens={xx: train_set_x})
    
    train_nll = theano.function(inputs = [index], updates=updates,outputs=cost,givens={xx:train_set_x[index*batch_size:(index+1)*batch_size],yy:train_set_y[index*batch_size:(index+1)*batch_size]})
    
    n_train_batchs = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    best_validation_loss = numpy.inf
    worse_time = 0
    stop = False
    epoch = 0
    vali_err = vali_error()
    test_err = test_error()
    print 'epoch 0 validation error:%f test error:%f'%(vali_err,test_err)
    start_time = time.clock()
    while (epoch < n_epochs) and not stop:
        epoch = epoch + 1
        for j in range(n_train_batchs):
            nll = train_nll(j)
        vali_err = vali_error()
        test_err = test_error()
        print 'epoch %d train error:%f validation error:%f test error:%f'%(epoch,nll,vali_err,test_err)
        if vali_err <= best_validation_loss + 0.0001:
            best_validation_loss = vali_err
            worse_time = 0
            best_hl_W = hl.W.get_value()
            best_hl_b = hl.b.get_value()
            best_W = logi.W.get_value()
            best_b = logi.b.get_value()
        else:
            worse_time += 1
            if worse_time == 30:
                hl.W.set_value(best_hl_W)
                hl.b.set_value(best_hl_b)
                logi.W.set_value(best_W)
                logi.b.set_value(best_b)
                stop = True;
    print 'bese validation error:%f test error:%f record:%f'%(vali_err,test_err,best_validation_loss)
    end_time = time.clock()
    print 'ran for %.1fs' % ((end_time - start_time))
    f = codecs.open('mlp.mat','wb')
    cPickle.dump(best_hl_W,f,2)  
    cPickle.dump(best_hl_b,f,2)
    cPickle.dump(best_W,f,2)  
    cPickle.dump(best_b,f,2)
    f.close()
    start_time = time.clock()   



def load_mlp(x,filename,n_hidden_layer,n_feature,n_class):
    logi = logi_re(x,n_feature,n_class)
    hl = hidden_layer(numpy.random.RandomState(1234),x,n_feature,n_class)
    f = codecs.open('logi.mat','rb')
    hl.W.set_value(cPickle.load(f)) 
    hl.b.set_value(cPickle.load(f))
    logi.W.set_value(cPickle.load(f))  
    logi.b.set_value(cPickle.load(f)) 
    f.close()
    return hl,logi

def load_re(x,filename,n_feature,n_class):
    logi = logi_re(x,n_feature,n_class)
    f = codecs.open('logi.mat','rb')
    logi.W.set_value(cPickle.load(f))  
    logi.b.set_value(cPickle.load(f)) 
    f.close()
    return logi

def own_write_regression(x,y,n_feature,n_class,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,with_regularization):
    '''
    datasets = load_data('mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    '''
    train_set_x = theano.shared(numpy.asarray(x[:train_part],dtype=theano.config.floatX),name='train_set_x')  # @UndefinedVariable
    train_set_y = T.cast(theano.shared(y[:train_part],name='train_set_y'),'int32')
    valid_set_x = theano.shared(numpy.asarray(x[validation_part:test_part],dtype=theano.config.floatX),name='valid_set_x')  # @UndefinedVariable
    valid_set_y = T.cast(theano.shared(y[validation_part:test_part],name='valid_set_y'),'int32')
    test_set_x = theano.shared(numpy.asarray(x[test_part:],dtype=theano.config.floatX),name='test_set_x')  # @UndefinedVariable
    test_set_y = T.cast(theano.shared(y[test_part:],name='test_set_y'),'int32')
    # accessing the third minibatch of the training set
    # Load the dataset
    xx = T.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    yy = T.ivector('y')
    index = T.lscalar()
    lamb = 0.01
    
    logi = logi_re(xx,n_feature,n_class)
    if with_regularization:
        cost = logi.nll(yy)+lamb*T.sqrt(T.sum(logi.W**2))
    else:
        cost = logi.nll(yy)

    w_grad = T.grad(cost, logi.W)
    b_grad = T.grad(cost, logi.b)
    
    updates = [(logi.W, logi.W-learning_rate*w_grad),(logi.b, logi.b-learning_rate*b_grad)]
    
    vali_error = theano.function(inputs = [], outputs=logi.error(yy),givens={xx: valid_set_x,yy: valid_set_y})
    
    test_error = theano.function(inputs = [], outputs=logi.error(yy),givens={xx: test_set_x,yy: test_set_y})
    
    test_result = theano.function(inputs = [], outputs=logi.pred_y, givens={xx: train_set_x})
    
    train_nll = theano.function(inputs = [index], updates=updates,outputs=cost,givens={xx:train_set_x[index*batch_size:(index+1)*batch_size],yy:train_set_y[index*batch_size:(index+1)*batch_size]})
    start_time = time.clock()
    n_train_batchs = train_set_x.get_value(borrow=True).shape[0] / batch_size
    for i in range(n_epochs):
        for j in range(n_train_batchs):
            nll = train_nll(j)
        vali_err = vali_error()
        test_err = test_error()
        print 'epoch %d train error:%f validation error:%f test error:%f'%(i,nll,vali_err,test_err)
    end_time = time.clock()
    f = codecs.open('logi.mat','wb')
    cPickle.dump(logi.W.get_value(),f,2)  
    cPickle.dump(logi.b.get_value(),f,2)
    f.close()
    print 'ran for %.1fs' % ((end_time - start_time))    
def own_write_regression_early_stopping(x,y,n_feature,n_class,train_part,validation_part,test_part,batch_size,n_epochs,learning_rate,with_regularization):
    '''
    datasets = load_data('mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    '''
    train_set_x = theano.shared(numpy.asarray(x[:train_part],dtype=theano.config.floatX),name='train_set_x')  # @UndefinedVariable
    train_set_y = T.cast(theano.shared(y[:train_part],name='train_set_y'),'int32')
    valid_set_x = theano.shared(numpy.asarray(x[validation_part:test_part],dtype=theano.config.floatX),name='valid_set_x')  # @UndefinedVariable
    valid_set_y = T.cast(theano.shared(y[validation_part:test_part],name='valid_set_y'),'int32')
    test_set_x = theano.shared(numpy.asarray(x[test_part:],dtype=theano.config.floatX),name='test_set_x')  # @UndefinedVariable
    test_set_y = T.cast(theano.shared(y[test_part:],name='test_set_y'),'int32')
    # accessing the third minibatch of the training set
    # Load the dataset
    xx = T.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    yy = T.ivector('y')
    index = T.lscalar()
    
    logi = logi_re(xx,n_feature,n_class)
    lamb = 0.01
    if with_regularization:
        cost = logi.nll(yy)+lamb*T.sqrt(T.sum(logi.W**2))
    else:
        cost = logi.nll(yy)

    w_grad = T.grad(cost, logi.W)
    b_grad = T.grad(cost, logi.b)
    
    updates = [(logi.W, logi.W-learning_rate*w_grad),(logi.b, logi.b-learning_rate*b_grad)]
    
    vali_error = theano.function(inputs = [], outputs=logi.error(yy),givens={xx: valid_set_x,yy: valid_set_y})
    
    test_error = theano.function(inputs = [], outputs=logi.error(yy),givens={xx: test_set_x,yy: test_set_y})
    
    test_result = theano.function(inputs = [], outputs=logi.pred_y, givens={xx: train_set_x})
    
    train_nll = theano.function(inputs = [index], updates=updates,outputs=cost,givens={xx:train_set_x[index*batch_size:(index+1)*batch_size],yy:train_set_y[index*batch_size:(index+1)*batch_size]})
    
    n_train_batchs = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    best_validation_loss = numpy.inf
    worse_time = 0
    stop = False
    epoch = 0
    vali_err = vali_error()
    test_err = test_error()
    print 'epoch 0 validation error:%f test error:%f'%(vali_err,test_err)
    start_time = time.clock()
    while (epoch < n_epochs) and not stop:
        epoch = epoch + 1
        for j in range(n_train_batchs):
            nll = train_nll(j)
        vali_err = vali_error()
        test_err = test_error()
        print 'epoch %d train error:%f validation error:%f test error:%f'%(epoch,nll,vali_err,test_err)
        if vali_err <= best_validation_loss + 0.0001:
            best_validation_loss = vali_err
            worse_time = 0
            best_W = logi.W.get_value()
            best_b = logi.b.get_value()
        else:
            worse_time += 1
            if worse_time == 200:
                logi.W.set_value(best_W)
                logi.b.set_value(best_b)
                stop = True;
    print 'bese validation error:%f test error:%f record:%f'%(vali_err,test_err,best_validation_loss)
    end_time = time.clock()
    print 'ran for %.1fs' % ((end_time - start_time))
    f = codecs.open('logi.mat','wb')
    cPickle.dump(best_W,f,2)  
    cPickle.dump(best_b,f,2)
    f.close()
    start_time = time.clock()
'''    
    patience = 2000
    patience_increase = 2
    epoch = 0
    best_validation_loss = numpy.inf
    improvement_threshold = 0.995
    stop = False
    validation_frequency = min(patience/2,n_train_batchs)
    while (epoch < n_epochs) and not stop:
        epoch = epoch + 1
        for j in range(n_train_batchs):
            nll = train_nll(j)
            vali_err = vali_error()
            test_err = test_error()
            print 'epoch %d %d train error:%f validation error:%f test error:%f'%(epoch,j,nll,vali_err,test_err)
            iter = (epoch-1)*(n_train_batchs)+j
            if (iter + 1) % validation_frequency == 0:
                if vali_err < best_validation_loss:
                    if vali_err < best_validation_loss * best_validation_loss:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = vali_err
            if patience <= iter:
                stop = True
                break    
    print 'best test error:%f'%(best_validation_loss)
    end_time = time.clock()
    print 'ran for %.1fs' % ((end_time - start_time))  
'''



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

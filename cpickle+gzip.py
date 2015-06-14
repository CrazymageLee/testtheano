import cPickle, gzip, numpy

if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    print test_set
    f.close()
    g = gzip.open('mini_mnist.pkl.gz','wb')
    cPickle.dump(test_set[0][:1200],g,2)
    cPickle.dump(test_set[1][:1200],g,2)
    g.close()
    f = gzip.open('mini_mnist.pkl.gz', 'rb')
    train_set = cPickle.load(f)
    print train_set
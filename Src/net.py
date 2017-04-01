# -*- coding: utf-8 -*-


import caffe
import numpy as np
from pylab import *
import csv
import scipy as sp

#Comment or uncomment following lines to set GPU mode

#caffe.set_mode_cpu()

caffe.set_device(0) # 0 correspond to the identification number of the GPU used
caffe.set_mode_gpu() # Chercher à utiliser les 2 GPU

# Prototxt directions
train_net_path = 'net_auto_train.prototxt'
test_net_path = 'net_auto_test.prototxt'
solver_config_path = 'net_auto_solver.prototxt'


'''
Net definition
'''
from caffe import layers as L, params as P

# Can be used directly
def Conv(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='xavier'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu
 



def lenet(lmdb, batch_size):

    n = caffe.NetSpec()
    # Input layer
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    # Residual convolution
    n.convres = L.Convolution(n.data, kernel_size=5, num_output=12,stride=1, weight_filler=dict(type='xavier'))
    # No activation for this first layer

    # Two layers of convolution
    n.conv1 = L.Convolution(n.convres, kernel_size=7, num_output=64,stride=2, weight_filler=dict(type='xavier'))
    n.batch_norm1 = L.BatchNorm(n.conv1, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.scale1 = L.Scale(n.batch_norm1, bias_term=True, in_place=True)
    n.relu2 = L.TanH(n.scale1, in_place=True)
    #n.relu2 = L.ReLU(n.scale1, in_place=True)
    n.pool1 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=48, stride=1, weight_filler=dict(type='xavier'))
    n.batch_norm2 = L.BatchNorm(n.conv2, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.scale2 = L.Scale(n.batch_norm2, bias_term=True, in_place=True)
    n.relu3 = L.TanH(n.scale2, in_place=True)
    #n.relu3 = L.ReLU(n.scale2, in_place=True)
    n.pool2 = L.Pooling(n.relu3, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # Dense classifier
    n.fc1 =   L.InnerProduct(n.pool2, num_output=4096, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.fc1, in_place=True)
    n.drop1 = L.Dropout(n.relu4, in_place=True)

    n.fc2 =   L.InnerProduct(n.drop1, num_output=4096, weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.fc2, in_place=True)
    n.drop2 = L.Dropout(n.relu5, in_place=True)

    # Outputs
    n.score = L.InnerProduct(n.drop2, num_output=2, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


# The net has been written to disk in a more verbose  but human-readable
# serialization format using Google's protobuf library.

def make_net():
    print 'Make train net'
    with open(train_net_path, 'w') as f:
        f.write(str(lenet('../Input/train_lmdb', 16)))
    print 'Make test net'
    with open(test_net_path, 'w') as f:
        f.write(str(lenet('../Input/validation_lmdb', 16)))

net = make_net()

'''
Define the solver
'''

from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

# Set a seed for reproducible experiments:
# this controls for randomization in training.
s.random_seed = 0xCAFFE

# Specify locations of the train and (maybe) test networks.
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 2000  # Test after every 500 training iterations.
s.test_iter.append(100) # Test on 100 batches each time we test.

s.max_iter = 10000     # no. of times to update the net (training iterations)
 
# EDIT HERE to try different solvers
# solver types include "SGD", "Adam", and "Nesterov" among others.
s.type = "SGD"

# Set the initial learning rate for SGD.
s.base_lr = 0.0001  # EDIT HERE to try different learning rates
# Set momentum to accelerate learning by
# taking weighted average of current and previous updates.
s.momentum = 0.9
# Set weight decay to regularize and prevent overfitting
s.weight_decay = 5e-4

# Set `lr_policy` to define how the learning rate changes during training.
# This is the same policy as our default LeNet.
s.lr_policy = 'inv'
s.gamma = 0.0001
s.power = 0.75
# EDIT HERE to try the fixed rate (and compare with adaptive solvers)
# `fixed` is the simplest policy that keeps the learning rate constant.
#s.lr_policy = 'fixed'

# Display the current training loss and accuracy every 1000 iterations.
#s.display = 1000

# Snapshots are files used to store networks we've trained.
# We'll snapshot every 5K iterations -- twice during training.
#s.snapshot = 5000
#s.snapshot_prefix = 'mnist/custom_net'

# Train on the GPU
s.solver_mode = caffe_pb2.SolverParameter.GPU

print 'Write solver'
# Write the solver to a temporary file and return its filename.
with open(solver_config_path, 'w') as f:
    f.write(str(s))

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
print 'Get solver'
solver = caffe.get_solver(solver_config_path)
print 'Solver init ok'

'''
Training loop
'''


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def normalize(nparray,alpha=-1):

    # Normalization of our first convolutional layer

    nparray = np.array(nparray)
    nparray[2,2] = alpha # Evaluate the influence of this alpha
    nparray = np.ma.array(nparray, mask=False)
    nparray.mask[2,2] = True  #Mask the center so he will not appear in the normalisation
    sumation = nparray.sum()
    nparray = nparray/sumation
    nparray = np.array(nparray) # return a nparray

    return nparray

def training_net(niter):
    niter = niter
    test_interval = niter/250

    # Losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter / test_interval)))


    # the main solver loop
    for it in range(niter):
        if it % 100 == 0:
            print 'Iteration number', it, 'on ' , niter

        ###### Set the first conv layer do derivate ######
        filters = solver.net.params['convres'][0].data[:,0]
        for i in range(12):
            filters[i] = normalize(filters[i],alpha=-1)
        solver.net.params['convres'][0].data[:,0] = filters
        
        ##### Solver batch and in train error evaluation
        solver.step(16)  # Number of images per batch (memory limitation of the GPU)

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            ll = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                               == np.int_(solver.test_nets[0].blobs['label'].data)) # Accuracy                               
                

            test_acc[it // test_interval] = correct / (100*16.) # batch_size = 16 and 100 batches at each test

    return test_acc, train_loss


[test_acc, train_loss] = training_net(7000)
print 'Test accuracy',  test_acc
print '---------------------------------------------------'

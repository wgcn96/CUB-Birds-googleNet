# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda0"
import theano
import lasagne
import pickle
import theano.tensor as T
import numpy as np

from collections import OrderedDict


import sys
script_path = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet'
sys.path.append(script_path)
from loadData import *
from googleNet import *
from function import *


np.random.seed(123)



#global Variant

LEARNING_RATE = 0.01
BATCH_SIZE = 12
DIM = 224
NUM_CLASSES = 200
NUM_EPOCHES = 300

# for debug
wangchendebug = 1
prefix = 'My debug output {}: '


print prefix.format(wangchendebug) + 'reload params from caffemodel...'
wangchendebug += 1

#copyParamsList = getOriParams(net_prototxt,net_weight)
reload_url = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/caffe_model.pkl'
copyParamsList = reloadModel(reload_url)
#copyParamsList[-2] = copyParamsList[-2].T
#################################################################


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1



net_stn = buildSTN_OneSub()
lasagneLayerList = lasagne.layers.get_all_layers(net_stn)
count1 = copyParams(lasagneLayerList[:82],copyParamsList)
count2 = copyParams(lasagneLayerList[86:169],copyParamsList)
#count3 = copyParams(lasagneLayerList[171:253],copyParamsList)

if (count1,count2) == (114,114):
    copyResult = True
else:
    copyResult = False

print prefix.format(wangchendebug) + 'build model and copy params finish!'
print "\tcopy params result: {}".format(copyResult)
print '\tcopy params are: {0}, {1}'.format(count1, count2)
wangchendebug += 1

getParamsList = lasagne.layers.get_all_params(net_stn, trainable=True)


X = T.tensor4()
y = T.ivector()

# training output
output_train = lasagne.layers.get_output([net_stn], X, deterministic=False)
output_train = output_train[0]
# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output([net_stn], X, deterministic=True)
output_eval = output_eval[0]
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
#updates = lasagne.updates.adam(cost, getParamsList, learning_rate=sh_lr)
#updates_finetune = lasagne.updates.adam(cost,getFineTuneParam, learning_rate= 10*sh_lr)

#train = theano.function([X, y], [cost, output_train], updates=updates)
#train_finetune = theano.function([X,y], [cost, output_train], updates=updates_finetune)


grads = []
for param in getParamsList:
    grad = T.grad(cost,param)
    grads.append(grad)
updates = OrderedDict()


for param,grad in zip(getParamsList[:120],grads[:120]):
    updates[param] = param - (1e-4)* sh_lr * grad
for param,grad in zip(getParamsList[120:234],grads[120:234]):
    updates[param] = param - sh_lr * grad
for param,grad in zip(getParamsList[234:],grads[234:]):
    updates[param] = param - 2 * sh_lr * grad

'''
updates = lasagne.updates.momentum(cost, getParamsList[:120], learning_rate= 1e-4 *sh_lr)

anotherupdates = []
anotherupdates.append( lasagne.updates.momentum(cost,getParamsList[120:234], learning_rate= sh_lr))
anotherupdates.append( lasagne.updates.momentum(cost,getParamsList[234:], learning_rate= 2 *sh_lr))
# anotherupdates.append( lasagne.updates.momentum(cost,getParamsList[171:252], learning_rate= sh_lr))
# anotherupdates.append( lasagne.updates.momentum(cost,getParamsList[252:], learning_rate= sh_lr))
for i in range(len(anotherupdates)):
    updates.update(anotherupdates[i])
'''


train = theano.function([X,y],[cost, output_train], updates=updates)
#eval = theano.function([X], [output_eval])
eval = theano.function([X], [output_eval], on_unused_input='warn')

def train_epoch(X, y):
    #X = X.get_value()
    num_samples = X.shape[0]
    shuffle = np.random.permutation(num_samples)     # 用random函数，打乱数据
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    costs = []
    correct = 0
    for i in range(num_batches):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        idx = shuffle[idx]
        X_batch = X[idx]
        y_batch = y[idx]
        cost_batch, output_train = train(X_batch, y_batch)
        costs += [cost_batch]
        preds = np.argmax(output_train, axis=-1)
        correct += np.sum(y_batch == preds)
    return np.mean(costs), correct / float(num_samples)


def eval_epoch(X, y):
    #X = X.get_value()
    num_samples = X.shape[0]
    shuffle = np.random.permutation(num_samples)
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    correct = 0
    for i in range(num_batches):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        idx = shuffle[idx]
        X_batch = X[idx]
        y_batch = y[idx]
        output_eval = eval(X_batch)
        preds = np.argmax(output_eval, axis=-1)
        correct += np.sum(y_batch == preds)
    acc = correct / float(num_samples)
    return acc

train_file_url = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTrainImages'
test_file_url = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTestImages'
data = makeDict(train_file_url,test_file_url,scaleLabel=False)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1


info = ''
loc_params = new_params = lasagne.layers.get_all_param_values(net_stn,trainable=True)
try:
    max_test_acc = 0.
    idx = 0
    #lrnStepSize = NUM_EPOCHES / 5
    lrnStepSize = 8
    for n in range(NUM_EPOCHES):

        train_cost, train_acc = train_epoch(data['X_train'], data['y_train'])
        test_acc = eval_epoch(data['X_test'], data['y_test'])

        loc_params = new_params
        new_params = lasagne.layers.get_all_param_values(net_stn,trainable=True)
        if n >= 1:
            print checkParams(loc_params,new_params)

        if (n+1) % lrnStepSize == 0:
            new_lr = sh_lr.get_value() * 0.96
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

        currentInfo = "Epoch {0}: Train cost {1}, Train acc {2}, test acc {3}".format(
                n, train_cost, train_acc, test_acc)
        info += currentInfo + '\n'
        print currentInfo


        if test_acc > max_test_acc:
            max_test_acc = test_acc
            idx = n+1
            model_url = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/stn_one_best_model.pkl'
            saveModel(model_url,lasagne.layers.get_all_param_values(net_stn,trainable = True))
            print "the model has been saved!"

    result = "we get best accuracy {0} , at the {1} epoch\n".format(max_test_acc,idx)
    result += "the model has been saved !"
    print result
    with open("/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/stn_one_result.txt", 'w') as f:
        f.write(info)
        f.write(result)
except KeyboardInterrupt:
    pass



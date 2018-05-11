# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda1"
import theano
import lasagne
import pickle
import theano.tensor as T
import numpy as np
import datetime

from collections import OrderedDict


import sys
script_path = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet'
sys.path.append(script_path)
from loadDataFast import *
from googleNet import *
from function import *

np.random.seed(123)


#global Variant

LEARNING_RATE = 0.02
BATCH_SIZE = 13
DIM = 224
NUM_CLASSES = 200
NUM_EPOCHES = 300

meanFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.npy'
meanNumpyArray = loadCaffeMean(meanFileURL)
model_url = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/best_model_2.pkl'
resultFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/result_2.txt'

# for record
info = ''
startTime = datetime.datetime.now()
info += 'start time is: ' + startTime.strftime('%Y-%m-%d %H:%M:%S') +'\n'
info += 'this experiment hyperparameters are:\n'
info += 'LEARNING_RATE = ' +str(LEARNING_RATE) +'\n'
info += 'BATCH_SIZE = '+str(BATCH_SIZE)+'\n'
info += 'NUM_EPOCHES = ' +str(NUM_EPOCHES)+'\n'
info += 'model_file :' + model_url +'\n'

##### if there are any things to record, write to str
info += '\n\t#####其余要说明的事项#####'
extra = '这次实验是使用了从磁盘读小批量数据，使用去均值操作，而没使用sklearn包。\n'
if extra != None:
    info += extra
else:
    info += 'None\n'
info += '\n'


# for debug
wangchendebug = 1
prefix = 'My debug output {}: '



train_file_url = '/home/wangchen/DataSet/CUB_200_2011/wgcnTrainImages'
test_file_url = '/home/wangchen/DataSet/CUB_200_2011/wgcnTestImages'
MyDict = makeDict(train_file_url, test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1


print prefix.format(wangchendebug) + 'reload params from caffemodel...'
wangchendebug += 1

reload_url = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/caffe_model.pkl'
copyParamsList = reloadModel(reload_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1


net_stn = buildSpatialTransformerNet()
lasagneLayerList = lasagne.layers.get_all_layers(net_stn)
count1 = copyParams(lasagneLayerList[:82],copyParamsList)
count2 = copyParams(lasagneLayerList[86:168],copyParamsList)
count3 = copyParams(lasagneLayerList[171:253],copyParamsList)

if (count1,count2,count3) == (114,114,114):
    copyResult = True
else:
    copyResult = False

print prefix.format(wangchendebug) + 'build model and copy params finish!'
print "\tcopy params result: {}".format(copyResult)
print '\tcopy params are: {0}, {1}, {2}'.format(count1, count2, count3)
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
for param,grad in zip(getParamsList[234:236],grads[234:236]):
    updates[param] = param - (1e-4)* sh_lr * grad
for param,grad in zip(getParamsList[236:],grads[236:]):
    updates[param] = param - sh_lr * grad





train = theano.function([X,y],[cost, output_train], updates=updates)
eval = theano.function([X], [output_eval], on_unused_input='warn')


def train_epoch(trainURL,trainLabel):
    num_samples = len(trainLabel)
    shuffle = np.random.permutation(num_samples)
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    costs = []
    correct = 0
    for i in range(num_batches):
        ndarray_idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        ndarray_idx = shuffle[ndarray_idx]
        X_batch,y_batch = loadBatchData(trainURL,trainLabel,ndarray_idx,meanNumpyArray)
        cost_batch, output_train = train(X_batch, y_batch)
        costs += [cost_batch]
        preds = np.argmax(output_train, axis=-1)
        correct += np.sum(y_batch == preds)
    return np.mean(costs), correct / float(num_samples)


def eval_epoch(testURL, testLabel):
    num_samples = len(testLabel)
    shuffle = np.random.permutation(num_samples)
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    correct = 0
    for i in range(num_batches):
        ndarray_idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        ndarray_idx = shuffle[ndarray_idx]
        X_batch,y_batch = loadBatchData(testURL,testLabel,ndarray_idx,meanNumpyArray)
        output_eval = eval(X_batch)
        preds = np.argmax(output_eval, axis=-1)
        correct += np.sum(y_batch == preds)
    acc = correct / float(num_samples)
    return acc

writeInfo(resultFileURL,info)
loc_params = new_params = lasagne.layers.get_all_param_values(net_stn,trainable=True)
try:
    max_test_acc = 0.
    idx = 0
    #lrnStepSize = NUM_EPOCHES / 5
    lrnStepSize = 8
    for n in range(NUM_EPOCHES):

        train_cost, train_acc = train_epoch(MyDict['trainURL'], MyDict['trainLabel'])
        test_acc = eval_epoch(MyDict['testURL'], MyDict['testLabel'])

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
        writeInfo(resultFileURL,currentInfo + '\n')
        print currentInfo

        if n <= 10:
            Mypause = 1

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            idx = n+1
            saveModel(model_url,lasagne.layers.get_all_param_values(net_stn,trainable = True))
            print "the model has been saved !"

    result = "we get best accuracy {0} , at the {1} epoch\n".format(max_test_acc,idx)
    result += "the model has been saved !"
    info = ''
    info += result+'\n'
    endTime = datetime.datetime.now()
    info += 'experiment end time is: ' + endTime.strftime('%Y-%m-%d %H:%M:%S') + '\n'
    totalSeconds = (endTime-startTime).seconds
    timeInfo =  'the program run total {} seconds .'.format(totalSeconds)
    info += timeInfo
    print result
    print timeInfo
    writeInfo(resultFileURL,info)
except KeyboardInterrupt:
    pass



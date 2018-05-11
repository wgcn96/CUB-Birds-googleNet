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

LEARNING_RATE = 0.02
BATCH_SIZE = 13
DIM = 224
NUM_CLASSES = 200
NUM_EPOCHES = 300

# for debug
wangchendebug = 1
prefix = 'My debug output {}: '


print prefix.format(wangchendebug) + 'reload params from the best model...'
wangchendebug += 1

reload_url = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/stn_4 params/best_model.pkl'
copyParamsList = reloadModel(reload_url)
'''
def getParamsValues(copyParamsList):
    valueList = []
    for i in range(len(copyParamsList)):
        valueList.append(copyParamsList[i].get_value())
    valueList = np.asarray(valueList)
    return valueList
copyParamsList = getParamsValues(copyParamsList)
'''

print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1


net_stn = buildSpatialTransformerNet()
lasagneLayerList = lasagne.layers.get_all_layers(net_stn)
lasagne.layers.set_all_param_values(net_stn,copyParamsList,trainable=True)
print prefix.format(wangchendebug) + 'build model and copy params finish!'
wangchendebug += 1

getParamsList = lasagne.layers.get_all_params(net_stn, trainable=True)


X = T.tensor4()
y = T.ivector()


valueLayerList = [lasagneLayerList[84],lasagneLayerList[169],lasagneLayerList[85],lasagneLayerList[170],lasagneLayerList[256]]
theta_1,theta_2,transform1,transform2,y_hat = lasagne.layers.get_output(valueLayerList,X,deterministic=True)
eval = theano.function([X], [theta_1,theta_2,transform1,transform2,y_hat], on_unused_input='warn')
print 'is it ok?'


def eval_input(data,data_y,n):
    count = 0
    for i in range(n):
        currentImage = data[i]
        currentImage = currentImage.reshape(1,3,224,224)
        y = data_y[i]
        theta_1, theta_2, transform1, transform2, y_hat = eval(currentImage)        # 由于输入的问题，输出的图片都是四维的，[0]
        print theta_1[0]
        showAnImage(transform1[0],mode='plt')
        print theta_2[0]
        showAnImage(transform2[0],mode='plt')
        y_hat = np.argmax(y_hat, axis=-1)
        print 'Ground truth is : {0}, the predicted label is : {1}'.format(y,y_hat)
        if y == y_hat:
            count += 1
    print 'total estimate {0} images, the truth times is {1}'.format(n,count)

data = makeDict(train_file_url,test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1

eval_input(data['X_test'],data['y_test'],10)




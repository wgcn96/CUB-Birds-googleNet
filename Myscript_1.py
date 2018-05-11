# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda1"
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



# for debug
wangchendebug = 1
prefix = 'My debug output {}: '
prefix_net = 'net/'

print prefix.format(wangchendebug) + 'reload params from caffemodel...'
wangchendebug += 1

prototxt = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/caffe_deploy_6.prototxt'
weightfile = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/CUB_googLeNet_ST_iter_4800.caffemodel'
copyParamsList = getOriParams(prototxt,weightfile)

net_stn = buildSpatialTransformerNet()
lasagneLayerList = lasagne.layers.get_all_layers(net_stn)
lasagneParamsList = lasagne.layers.get_all_param_values(net_stn)
'''
count1 = copyParams(lasagneLayerList[:82],copyParamsList)
count2 = copyParams(lasagneLayerList[86:168],copyParamsList)
count3 = copyParams(lasagneLayerList[171:253],copyParamsList)
'''

checkDimes(lasagneParamsList,copyParamsList)





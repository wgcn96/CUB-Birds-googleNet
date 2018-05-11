# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda0"
import theano
import lasagne
import pickle
import theano.tensor as T
import numpy as np
import datetime
from collections import OrderedDict
import json
import csv


import sys
script_path = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet'
sys.path.append(script_path)
from loadData import *
from googleNet import *
from function import *


# for debug
wangchendebug = 1
prefix = 'My debug output {}: '
net_prefix = 'google/'

data = makeDict(train_file_url,test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1

print prefix.format(wangchendebug) + 'reload params from the best model...'
wangchendebug += 1

reload_url = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/67.8%/copy_param_best_model2.pkl'
copyParamsList = reloadModel(reload_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1


#net_vgg_Dict = build_vgg_model(prefix=net_prefix,inputLayer=net_input,classificationFlag=True)
#net_vgg_last_layer = net_vgg_Dict[net_prefix + 'prob']

input = InputLayer((None, 3, 224, 224), name=net_prefix+'input')
net_googleNet = build_model(input=input,prefix=net_prefix,lastClassNum=200,dropoutratio=0.4,classification=True)
net_googleNet_last_layer = net_googleNet[net_prefix + 'prob']

lasagneLayerList = lasagne.layers.get_all_layers(net_googleNet_last_layer)
lasagne.layers.set_all_param_values(net_googleNet_last_layer,copyParamsList,trainable=True)

print prefix.format(wangchendebug) + 'build model and copy params finish!'
wangchendebug += 1


X = T.tensor4()
y = T.ivector()

y_hat = lasagne.layers.get_output(net_googleNet_last_layer,X,deterministic=True)
eval = theano.function([X],[y_hat])


#------------------
# data : 图片数据
# data_y : 标签数据
# N ： 测试数据总数
# n ： 批量测试大小（可改为1 或其它值）
#------------------

def eval_input(data,data_y,N,n=10):
    count = 0.
    shuffle = np.asarray(range(N))  # np.random.permutation(N)     # 用random函数，打乱数据
    step = int(np.ceil(N/float(n)))
    all_pred = []
    all_probs = []
    for i in range(step):
        idx = range(i * n, np.minimum((i + 1) * n, N))
        pos = shuffle[idx]
        currentImage = data[pos]
        y = data_y[pos]
        y_hat = eval(currentImage)
        for hat in y_hat[0]:
            all_probs.append(hat)
        # all_probs += y_hat[0]
        preds = np.argmax(y_hat, axis=-1)
        # print preds.shape
        all_pred += preds[0].tolist()
        count += np.sum(y == preds)
    # for i in all_pred:
    #     print i
    print len(all_probs), len(all_probs[0])
    accu = count/N
    result =  'total estimate {0} images, the truth times is {1}, accu is: {2}'.format(N,count,accu)
    print result
    return all_pred, all_probs


print test_file_url
all_pred, all_probs = eval_input(data['X_test'],data['y_test'],1429)
all_name = []
# eval_input(data['X_train'],data['y_train'],5994)
with open(test_file_url) as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        if len(arr) < 2:
            continue
        pic_name = arr[0].split('/')[-1]
        # print pic_name
        all_name.append(pic_name)
all_pred = (np.asarray(all_pred) + 1).tolist()
d = {}
for i, j in zip(all_name, all_pred):
    value = str(j).zfill(3)
    d[i] = value
print all_pred
print d
with open('/home/wangchen/xxt/xaircraft_exam/train3/predict.json', 'wb') as json_file:
    json_file.write('{}\n'.format(json.dumps(d, indent=4)))
with open('/home/wangchen/xxt/xaircraft_exam/train3/prob.csv', 'wb') as prob_file:
    csv_writer = csv.writer(prob_file)
    for i, j in zip(all_name, all_probs):
        j = j.tolist()
        j.insert(0, i)
        print j[0], len(j)
        csv_writer.writerow(j)


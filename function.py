# -*- coding: UTF-8 -*-


import pickle
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import cv2

from collections import OrderedDict

def reloadModel(url):
    with open(url, 'rb') as f:
        paramList = pickle.load(f)
    return paramList

def saveModel(url,params):
    with open(url, 'wb') as f:
        pickle.dump(params, f)

def copyParams(lasagneLayerList,oriParamList):
    n = len(lasagneLayerList)
    pos = 0
    for i in range(n):
        currentLayer = lasagneLayerList[i]
        currentParamList = currentLayer.get_params(trainable=True)
        if currentParamList != []:
            for param in currentParamList:
                param.set_value(oriParamList[pos])
                pos += 1
    return pos          #处理的參數數量


def checkParams(valueList1, valueList2):
    resultCounter = 0
    for i in range(len(valueList1)):
        param1 = valueList1[i]
        param2 = valueList2[i]
        if (param1 == param2).all() == True:
            resultCounter += 1
    if resultCounter == 0:
        return True,resultCounter
    else:
        return False,resultCounter

def showAnImage(imageData,mode='cv2',label='0',waitKey=0):
    revortedImage = np.transpose(imageData,(1,2,0))
    revortedImage = cv2.cvtColor(revortedImage, cv2.COLOR_BGR2RGB)
    if mode == 'cv2':
        cv2.imshow(label,revortedImage)
        cv2.waitKey(waitKey)
        cv2.destroyAllWindows()
    else:
        plt.imshow(revortedImage)
        plt.show()

def writeInfo(fileURL,info):
    with open(fileURL,'a') as f:
        f.write(info)
    f.close()


'''
######import caffe
caffe_root = '/home/wangchen/last_caffe_with_stn-master/python'
import sys
sys.path.insert(0,caffe_root)
import caffe
print caffe.__path__


def getOriParams(net_prototxt,net_weight):
    net_ori = caffe.Net(net_prototxt, net_weight, caffe.TEST)
    paramList = []
    for param_name in net_ori.params.keys():
        weight = net_ori.params[param_name][0].data
        bias = net_ori.params[param_name][1].data
        paramList.append(weight)
        paramList.append(bias)
        #print param_name,weight,bias
    return paramList


def checkDimes(paramsList1_theano, paramsList2_caffe):
    print 'paramsList1.dimes is: {0}'.format(len(paramsList1_theano))
    print 'paramsList2.dimes is: {0}'.format(len(paramsList2_caffe))
    for i in range(len(paramsList1_theano)):
        result = ''
        #result +=  paramsList1_theano[i].name
        theano_shape = paramsList1_theano[i].shape
        caffe_shape = paramsList2_caffe[i].shape
        if theano_shape == caffe_shape:
            result += '\tTrue'
        else:
            result += '\tFalse'
            result += '\n'
            result += str(i) + '\t'
            result += 'caffe_dimes is: {0} ,target_dims is: {1}'.format(caffe_shape,theano_shape)
        print result



def convertCaffeMeanFile(meanFileURL, npyFileURL):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(meanFileURL, 'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean = arr[0]
    np.save(npyFileURL, npy_mean)
    print 'finish!'
'''

if __name__ == '__main__':
    #npyFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.npy'
    #meanFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.binaryproto'
    #convertCaffeMeanFile(meanFileURL,npyFileURL)

    info1 = 'info1'
    info2 = 'info2'
    testtxt = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/test.txt'
    writeInfo(testtxt,info1)
    writeInfo(testtxt,info2)
    print 'finish!'
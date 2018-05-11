# -*- coding: UTF-8 -*-

##### 一次将图片加载至内存

import string
import cv2
import numpy as np
import lasagne

from sklearn import preprocessing

# global variance
train_file_url = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTrainImages'
test_file_url = '/home/wangchen/xxt/xaircraft_exam/train3/test_url.txt'  # '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTestImages'

# script
# 将图片的标签改为从0开始
def changeLabel(ori_root,new_root,oriFileURL,newFileURL):
    oriFile = open(oriFileURL,'r')
    newFile = open(newFileURL,'w')
    while True:
        line = oriFile.readline()
        if line == '':
            break
        line = line.strip()
        pos = line.find(' ')
        url = line[:pos]
        labelstr = line[pos+1:]
        newurl = url.replace(ori_root,new_root)
        labelstr = str(string.atoi(labelstr) - 1)
        newurl = newurl + ' ' + labelstr + '\n'
        newFile.write(newurl)
    oriFile.close()
    newFile.close()

def loadImage(imageURL):
    ima = cv2.imread(imageURL)
    ima = cv2.resize(ima, (224, 224))  # 图像像素调整 ——》224*224
    ima = np.asarray(ima, dtype='float32') / 255.
    #cv2.cvtColor()
    ima = ima.transpose(2, 0, 1)  # 这张图片的格式为(h,w,rgb), 然后想办法交换成(rgb,h,w)
    return ima


def loadImageFromFile(fileURL):
    data = []
    label = []
    file = open(fileURL,'r')
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip()
        pos = line.find(' ')
        image_url = line[:pos]
        label_str = line[pos + 1:]
        curLabel = string.atoi(label_str)
        curIma = loadImage(image_url)    # 加载单张图片，用append方法拼成数组
        data.append(curIma)
        label.append(curLabel)
    return data,label


def scaleData(train_data, test_data):
    m = len(train_data)
    n = len(test_data)
    train_data = np.array(train_data).ravel()
    train_data = train_data.reshape(train_data.shape[0],1)
    test_data = np.array(test_data).ravel()
    test_data = test_data.reshape(test_data.shape[0],1)
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    train_data = np.reshape(train_data,(m,3,224,224))
    test_data = np.reshape(test_data,(n,3,224,224))
    return train_data,test_data

# 将转换后的图像显示出来，以验证正确性
# debug用
def checkAnImage(transpose_image):
    image = np.transpose(transpose_image,(1,2,0))
    cv2.imshow("image", image)
    cv2.waitKey(0)

'''
def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = T._shared(
        np.asarray(data_x, dtype=theano.config.floatX),
        borrow=borrow)
    shared_y = T._shared(
        np.asarray(data_y,dtype=theano.config.floatX),
        borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')
'''

def makeDict(train_file_url,test_file_url,scaleLabel=True):
    trainData, trainLabel = loadImageFromFile(train_file_url)  # list list
    testData, testLabel = loadImageFromFile(test_file_url)  # list list
    if scaleLabel:
        trainData, testData = scaleData(trainData, testData)  # ndarray ndarray
    X_train = lasagne.utils.floatX(trainData)
    y_train = np.array(trainLabel)
    y_train = y_train.astype('int32')
    X_test = lasagne.utils.floatX(testData)
    y_test = np.array(testLabel)
    y_test = y_test.astype('int32')
    '''
    X_train = T._shared(np.asarray(X_train,dtype=theano.config.floatX),borrow=True)
    X_test = T._shared(np.asarray(X_test,dtype=theano.config.floatX),borrow=True)
    y_train = T._shared(np.asarray(y_train,dtype=theano.config.floatX),borrow=True)
    y_test = T._shared(np.asarray(y_test,dtype=theano.config.floatX),borrow=True)
    y_train = T.cast(y_train, 'int32')
    y_test = T.cast(y_test, 'int32')
    '''
    return dict(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

if __name__ == '__main__':
    '''
    ori_root = '/media/lsq/data2/CUB_200_2011'
    new_root = '/media/wangchen/newdata1/wangchen/dataSet'
    oriFile = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTestImages'
    newFile = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTestImages_3'
    changeLabel(ori_root,new_root,oriFile,newFile)
    '''

    '''
    trainData, trainLabel = loadFile(train_file_url)
    testData, testLabel = loadFile(test_file_url)
    trainData, testData = scaleData(trainData,testData)
    '''
    data = makeDict(train_file_url,test_file_url)
    print 'check finish!'


# -*- coding: UTF-8 -*-

##### 一次将图片加载至内存

import string
import cv2
import lasagne
import numpy as np


from sklearn import preprocessing

meanFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.npy'



def loadCaffeMean(meanFileURL):
    mu = np.load(meanFileURL)
    # mu = mu.mean(1).mean(1)
    # print 'mean-subtracted values:', zip('BGR', mu)
    return mu.astype('float32')


def loadImage(imageURL,meanNumpyArray):
    ima = cv2.imread(imageURL)          #使用cv2 读入图片，bgr格式
    ima = cv2.resize(ima, (224, 224))  # 图像像素调整 ——》224*224
    ima = np.asarray(ima,dtype='float32')
    ima = ima.transpose(2, 0, 1)  # 这张图片的格式为(h,w,c), 然后想办法交换成(c,h,w)
    ima -= meanNumpyArray           # 去均值操作
    ima = ima / 255.

    return ima


def loadURLFromFile(fileURL):
    url = []
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
        curLabel = string.atoi(label_str)       #int
        url.append(image_url)
        label.append(curLabel)
    return url,label

'''
def makeDict(train_file_url,test_file_url):
    trainData, trainLabel = loadImageFromFile(train_file_url)  # list list
    testData, testLabel = loadImageFromFile(test_file_url)  # list list
    X_train = lasagne.utils.floatX(trainData)
    y_train = np.array(trainLabel)
    y_train = y_train.astype('int32')
    X_test = lasagne.utils.floatX(testData)
    y_test = np.array(testLabel)
    y_test = y_test.astype('int32')
    return dict(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
'''


def makeDict(train_file_url,test_file_url):
    trainURL , trainLabel = loadURLFromFile(train_file_url)
    testURL  , testLabel = loadURLFromFile(test_file_url)
    return dict(trainURL=trainURL,trainLabel=trainLabel,testURL=testURL,testLabel=testLabel)

def loadBatchData(URL_list, Label_list, ndarry_idx, meanNumpyArray):
    idx = ndarry_idx.tolist()
    X = []    # list
    y = []    # list
    for i in range(len(idx)):
        current_idx = idx[i]
        current_url = URL_list[current_idx]
        current_label = Label_list[current_idx]
        current_image = loadImage(current_url,meanNumpyArray)
        X.append(current_image)
        y.append(current_label)
    X = lasagne.utils.floatX(X)     #covert to ndarray
    y = np.array(y,dtype='int32')                   #covert to ndarray
    return X,y

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
        newurl = newurl + ' ' + labelstr + '\n'
        newFile.write(newurl)
    oriFile.close()
    newFile.close()

# 将转换后的图像显示出来，以验证正确性
# debug用
def checkAnImage(transpose_image):
    image = np.transpose(transpose_image,(1,2,0))
    cv2.imshow("image", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    '''
    ori_root = '/media/wangchen/newdata1/wangchen/'
    new_root = '/home/wangchen/'
    oriFile = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTrainImages'
    newFile = '/home/wangchen/DataSet/CUB_200_2011/wgcnTrainImages_2'
    changeLabel(ori_root,new_root,oriFile,newFile)
    '''

    '''
    train_file_url = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTrainImages'
    test_file_url = '/media/wangchen/newdata1/wangchen/dataSet/CUB_200_2011/wgcnTestImages'
    data = makeDict(train_file_url,test_file_url)
    '''

    imageURL = '/home/wangchen/DataSet/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
    meanFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.npy'
    meanNumpyArray = loadCaffeMean(meanFileURL)
    ima = loadImage(imageURL,meanNumpyArray)
    checkAnImage(ima)
    print 'check finish!'


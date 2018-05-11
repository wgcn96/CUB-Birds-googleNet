# -*- coding: UTF-8 -*-

# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear, sigmoid
import lasagne
import numpy as np


def build_inception_module(name, prefix, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    subnet = {}
    subnet['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1, name=prefix+name+'/pool')
    subnet['pool_proj'] = ConvLayer(
        subnet['pool'], nfilters[0], 1, flip_filters=False, name=prefix+name+'/pool_proj')

    subnet['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False, name=prefix+name+'/1x1')

    subnet['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False, name=prefix+name+'/3x3_reduce')
    subnet['3x3'] = ConvLayer(
        subnet['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False, name=prefix+name+'/3x3')

    subnet['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False, name=prefix+name+'/5x5_reduce')
    subnet['5x5'] = ConvLayer(
        subnet['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False, name=prefix+name+'/5x5')

    subnet['output'] = ConcatLayer([
        subnet['1x1'],
        subnet['3x3'],
        subnet['5x5'],
        subnet['pool_proj'],
        ], name=prefix+name+'/output')

    return {prefix+'{}/{}'.format(name, k): v for k, v in subnet.items()}


def build_model(input, prefix, lastClassNum=200, dropoutratio=0.4, classification=False):
    net = {}

    net[prefix+'input'] = input

    net[prefix+'conv1/7x7_s2'] = ConvLayer(
        net[prefix+'input'], 64, 7, stride=2, pad=3, flip_filters=False, name=prefix+'conv1/7x7_s2')

    net[prefix+'pool1/3x3_s2'] = PoolLayer(
        net[prefix+'conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False, name=prefix+'pool1/3x3_s2')

    net[prefix+'pool1/norm1'] = LRNLayer(net[prefix+'pool1/3x3_s2'], alpha=0.00002, k=1, name=prefix+'pool1/norm1')

    net[prefix+'conv2/3x3_reduce'] = ConvLayer(
        net[prefix+'pool1/norm1'], 64, 1, flip_filters=False, name=prefix+'conv2/3x3_reduce')

    net[prefix+'conv2/3x3'] = ConvLayer(
        net[prefix+'conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False, name=prefix+'conv2/3x3')

    net[prefix+'conv2/norm2'] = LRNLayer(net[prefix+'conv2/3x3'], alpha=0.00002, k=1, name=prefix+'conv2/norm2')

    net[prefix+'pool2/3x3_s2'] = PoolLayer(
      net[prefix+'conv2/norm2'], pool_size=3, stride=2, ignore_border=False, name=prefix+'pool2/3x3_s2')

    net.update(build_inception_module('inception_3a',
                                      prefix,
                                      net[prefix+'pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]), )
    net.update(build_inception_module('inception_3b',
                                      prefix,
                                      net[prefix+'inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net[prefix+'pool3/3x3_s2'] = PoolLayer(
      net[prefix+'inception_3b/output'], pool_size=3, stride=2, ignore_border=False, name=prefix+'pool3/3x3_s2')

    net.update(build_inception_module('inception_4a',
                                      prefix,
                                      net[prefix+'pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      prefix,
                                      net[prefix+'inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      prefix,
                                      net[prefix+'inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      prefix,
                                      net[prefix+'inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      prefix,
                                      net[prefix+'inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net[prefix+'pool4/3x3_s2'] = PoolLayer(
      net[prefix+'inception_4e/output'], pool_size=3, stride=2, ignore_border=False, name=prefix+'pool4/3x3_s2')

    net.update(build_inception_module('inception_5a',
                                      prefix,
                                      net[prefix+'pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      prefix,
                                      net[prefix+'inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net[prefix+'pool5/7x7_s1'] = GlobalPoolLayer(net[prefix+'inception_5b/output'], name=prefix+'pool5/7x7_s1')

    net[prefix+'dropout'] = lasagne.layers.DropoutLayer(net[prefix+'pool5/7x7_s1'], p=dropoutratio, name=prefix+'dropout')

    if classification == True:
        net[prefix+'loss3/classifier'] = DenseLayer(net[prefix+'dropout'],
                                             num_units=lastClassNum,
                                             nonlinearity=linear,
                                             name=prefix+'loss3/classifier')
        net[prefix+'prob'] = NonlinearityLayer(net[prefix+'loss3/classifier'],
                                        nonlinearity=softmax, name=prefix+'prob')
    return net


def buildSTN(input, stn_downsample_factor=1):

    st_prefix = 'st/'

    b1 = [0.5 , -0.5,  0.5, 0]
    b2 = [0.5 , 0.5 , 0.5, 0]
    b1 = np.asarray(b1,dtype='float32')
    b2 = np.asarray(b2,dtype='float32')

    googleNetDict_st = build_model(input, prefix='st/')

    net_st_pre = googleNetDict_st['st/inception_5b/output']
    net_input = googleNetDict_st['st/input']

    net_st_final_conv = ConvLayer(net_st_pre,num_filters=128,filter_size=1,name=st_prefix+'final_conv')

    net_st_pre_theta_fc128 = DenseLayer(net_st_final_conv,128,name=st_prefix+'final_fc')

    net_st_theta_1_fc = DenseLayer(net_st_pre_theta_fc128,4,name=st_prefix+'theta_1',b=b1, W=lasagne.init.Constant(0.0),nonlinearity=None)

    net_st_theta_2_fc = DenseLayer(net_st_pre_theta_fc128,4,name=st_prefix+'theta_2',b=b2, W=lasagne.init.Constant(0.0),nonlinearity=None)

    net_stn_1 = lasagne.layers.TransformerLayer(net_input, net_st_theta_1_fc,mode=4, name=st_prefix+'stn_1',downsample_factor=stn_downsample_factor)

    net_stn_2 = lasagne.layers.TransformerLayer(net_input, net_st_theta_2_fc,mode=4, name=st_prefix+'stn_2',downsample_factor=stn_downsample_factor)

    #print "Transformer network output shape: ", net_stn_1.output_shape
    return net_stn_1,net_stn_2

def buildSpatialTransformerNet():

    st_prefix = 'st/'
    inc1_prefix = 'inc1/'
    inc2_prefix = 'inc2/'

    input = InputLayer((None, 3, 224, 224), name=st_prefix+'input')
    net_stn_1, net_stn_2 = buildSTN(input=input)

    googleNetDict_inc1 = build_model(input=net_stn_1,prefix=inc1_prefix,dropoutratio=0.7)
    googleNetDict_inc2 = build_model(input=net_stn_2,prefix=inc2_prefix,dropoutratio=0.7)

    net_inc1_output = googleNetDict_inc1[inc1_prefix+'dropout']
    net_inc2_output = googleNetDict_inc2[inc2_prefix+'dropout']

    net_final_concat = ConcatLayer([net_inc1_output,net_inc2_output], name='final/concat/output')

    net_final_fc = DenseLayer(net_final_concat,200,nonlinearity=linear,
                                         name='final/fc')

    net_final_prob = NonlinearityLayer(net_final_fc,
                                    nonlinearity=softmax, name='final/prob')

    return net_final_prob


def buildSTN_OneSub():
    st_prefix = 'st/'
    inc1_prefix = 'inc1/'
    #inc2_prefix = 'inc2/'

    input = InputLayer((None, 3, 224, 224), name=st_prefix+'input')
    net_stn_1, net_stn_2 = buildSTN(input=input)

    googleNetDict_inc1 = build_model(input=net_stn_1,prefix=inc1_prefix,dropoutratio=0.5)
    #googleNetDict_inc2 = build_model(input=net_stn_2,prefix=inc2_prefix,dropoutratio=0.5)

    net_inc1_output = googleNetDict_inc1[inc1_prefix+'dropout']
    #net_inc2_output = googleNetDict_inc2[inc2_prefix+'dropout']

    #net_final_concat = ConcatLayer([net_inc1_output,net_inc2_output], name='final/concat/output')

    net_final_fc = DenseLayer(net_inc1_output,200,nonlinearity=linear,
                                         name='final/fc')

    net_final_prob = NonlinearityLayer(net_final_fc,
                                    nonlinearity=softmax, name='final/prob')

    return net_final_prob


if __name__ == '__main__':
    net_stn  = buildSpatialTransformerNet()
    net_stn_one_sub = buildSTN_OneSub()
    print 'finish!'

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:39:19 2019

@author: saugata paul

This python file is used to correctly determine upto which layer numbers the 
layers can be fixed. The way this should work is as follows:
    
    1. load the model weights from the desired location
    2. pass the model to plot_layer_arch(). now you can
       see the model architecture and visually determine
       which layers you want to freeze. look at the archi
       -tecture and see for it yourself the layer name
       upto which you want the model to have freezed layers.
    3. once you have the layer name, get the layer index by
       making a function call to get_layer_index()
    4. now we have a layer index value. go back to train_
       pipeline.py and set the corresponding upto this 
       layer index to trainable=false, the rest of the layers
       will be set to trainabale=True      

"""

from keras.applications import vgg16, xception, resnet, inception_v3, inception_resnet_v2, nasnet
from keras.utils import plot_model

weights_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/weights/"
model_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/models/"


def plot_layer_arch(model, model_name, stage_no):
    """
    Get the model architecture, so that you can make
    the decision upto which layers you can freeze.
    The particular layer name can in inferred from 
    this plot.
    """
    plot_model(model, to_file=model_path+'{}_model_architecture_stage_{}.pdf'.format(model_name,stage_no), show_shapes=True, show_layer_names=True)

def get_layer_index(model,layer_name):
    """
    Once you get the layer name, you can get the cores-
    ponding layer number. You can freeze all the layers
    before this number
    """
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    print("Freeze layers upto this index: ",layer_idx)
    
def get_all_layer_names(model):
    """
    This function is get the names of all the layers
    that are there in the model
    """
    layers_dic=dict()
    for i in range(0,len(xcp.layers)):
        layers_dic[i]=xcp.layers[i]
    
    print(layers_dic.items())


vgg=vgg16.VGG16(weights=None, include_top=False)
vgg.load_weights(weights_path+"vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

incv3 = inception_v3.InceptionV3(weights=None, include_top=False)
incv3.load_weights(weights_path+"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

res50 = resnet.ResNet50(weights=None, include_top=False)
res50.load_weights(weights_path+"resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

incres = inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False)
incres.load_weights(weights_path+"inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")

nas = nasnet.NASNetLarge(weights=None, include_top=False)
nas.load_weights(weights_path+"NASNet-large-no-top.h5")

xcp = xception.Xception(weights=None, include_top=False)
xcp.load_weights(weights_path+"xception_weights_tf_dim_ordering_tf_kernels_notop.h5")


model=incres
model_name="inc_res"
stage_no=2
layer_name="conv2d_189"
    
plot_layer_arch(model, model_name, stage_no)

get_layer_index(model,layer_name)



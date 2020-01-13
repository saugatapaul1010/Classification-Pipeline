#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:52:08 2020

@author: saugata paul
"""
import pandas as pd
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from keras.applications.resnet import preprocess_input as preprocess_input_resnet50
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications import vgg16, xception, resnet, inception_v3, inception_resnet_v2, nasnet

class Utility:
    
    def __init__(self, input_params, path_dict):
        self.input_params = input_params
        self.path_dict = path_dict
    
    def load_data(self, set_name):
        """
        This function is used to load the training, 
        validation as well as the test data.
        
        3 datasets are present => train.msgpack, val.msgpack, test.msgpack
        """

        dataframe = pd.read_msgpack(self.path_dict['df_path']+"{}.msgpack".format(set_name))
        
        return dataframe

    def init_sizes(self):
        """
        This block of code is used to initialize the input sizes
        of the images for specific models. Because specific models
        are trained using images of specific sizes. If any new 
        model is added, their corresponding input sizes has to be
        placed here.
        """
        size_dict = dict()
        size_dict["vgg16"] = (224, 224)
        size_dict["inceptionv3"] = (299, 299)
        size_dict["resnet50"] = (224, 224)
        size_dict["inception_resnet"] = (299, 299)
        size_dict["nasnet"] = (331, 331)
        size_dict["xception"] = (299, 299)
        
        return size_dict[self.input_params['model_name']]
    
    def init_preprocess_func(self):
        """
        This block of code is used to initialize the input sizes
        of the images for specific models. Because specific models
        are trained using images of specific sizes. If any new 
        model is added, their corresponding input sizes has to be
        placed here.
        """
        pre_func = dict()
        pre_func["vgg16"] = preprocess_input_vgg16
        pre_func["inceptionv3"] = preprocess_input_inceptionv3
        pre_func["resnet50"] = preprocess_input_resnet50
        pre_func["inception_resnet"] = preprocess_input_inception_resnet_v2
        pre_func["nasnet"] = preprocess_input_nasnet
        pre_func["xception"] = preprocess_input_xception
        
        return pre_func[self.input_params['model_name']]

    def get_models(self, stage_no):
        """
        This function is used to load the saved keras models.
        The models will be loaded based on the model type name
        and the training stage.
        
        Arguments:                    
    
            -model_name : Name of the model, for example - vgg16, inception_v3, resnet50 etc
            
            -stage_no   : The stage of training for which the evaluation has tp be done. This pipeline 
                          is trained in two stages 1 and 2. The stage number is needed to save the 
                          architecture for individual stages and have unique file names. 
        """
        model = load_model(self.path_dict["model_path"]+"stage{}/".format(stage_no)+"{}_model_stage_{}.h5".format(self.input_params["model_name"], stage_no))
        
        return model
    
    def load_imagenet_model(self):
        """
        Initialize the pre-trained model architecture and load the model weights.
        The downloaded weights contains only the convolution base. It does not
        contain the top two dense layers. We will have to manually define the top
        two dense layers. The size_dict dictionary object will hold the input sizes
        for various models, which will be further used to train the respective models
        with the given input image dimensions.
        
        Arguments:                    
    
            -model_name : Name of the model, for example - vgg16, inception_v3, resnet50 etc
    
        """
    
        if(self.input_params['model_name']=="vgg16"):
            base_model = vgg16.VGG16(weights=None, include_top=False)
            base_model.load_weights(self.path_dict["weights_path"]+"vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        elif(self.input_params['model_name']=="inceptionv3"):
            base_model = inception_v3.InceptionV3(weights=None, include_top=False)
            base_model.load_weights(self.path_dict["weights_path"]+"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
        elif(self.input_params['model_name']=="resnet50"):
            base_model = resnet.ResNet50(weights=None, include_top=False)
            base_model.load_weights(self.path_dict["weights_path"]+"resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
        elif(self.input_params['model_name']=="inception_resnet"):
            base_model = inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False)
            base_model.load_weights(self.path_dict["weights_path"]+"inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")
        elif(self.input_params['model_name']=="nasnet"):
            base_model = nasnet.NASNetLarge(weights=None, include_top=False)
            base_model.load_weights(self.path_dict["weights_path"]+"NASNet-large-no-top.h5")
        elif(self.input_params['model_name']=="xception"):
            base_model = xception.Xception(weights=None, include_top=False)
            base_model.load_weights(self.path_dict["weights_path"]+"xception_weights_tf_dim_ordering_tf_kernels.h5")
        return base_model
        
    def no_of_classes(self):
        """
        This function will be determine the number of classes that
        the model needs to train on. This function will determine
        the number of classes automatically without the user having
        to input the number of classes manually.
        """
        
        df = pd.read_msgpack(self.path_dict['df_path']+"train.msgpack")
        classes = df["class_label"].nunique()
        
        return classes
        
        
        
        
        
        
        
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 19:18:20 2019

@author: saugata paul
"""

#Import the deep learning libraries
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.callbacks import History
from keras.applications import vgg16, xception, resnet, inception_v3, inception_resnet_v2, nasnet
import pandas as pd
import os
from datetime import datetime as dt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
import evaluation
from keras.utils import plot_model
from keras.applications.vgg16 import preprocess_input
from contextlib import redirect_stdout

#FORCE GPU USE
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5 --no-check-certificate

def load_data(set_name, path_dict):
    """
    This function is used to load the training, 
    validation as well as the test data.
    
    3 datasets are present => train.msgpack, val.msgpack, test.msgpack
    """

    df = pd.read_msgpack(path_dict['df_path']+"{}.msgpack".format(set_name))
    return df

def plot_layer_arch(model, model_name, stage_no, path_dict):
    """
    Get the model architecture, so that you can make
    the decision upto which layers you can freeze.
    The particular layer name can in inferred from
    this plot.
    
    Arguments:
        
        -model      : The trained model
                      
        -model_name : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                      
        -stage_no   : The stage of training. This pipeline is trained in two stages 1 and 2. 
                      The stage number is needed to save the architecture for individual stages
                      and have unique file names
    """
    plot_model(model, to_file=path_dict['model_path']+"stage{}/".format(stage_no)+'{}_model_architecture_stage_{}.pdf'.format(model_name,stage_no), show_shapes=True, show_layer_names=True)

def save_summary(model, model_name, stage_no, path_dict):
    """
    This function is used to save the model summary along with 
    the number of parameters at each stage.
    
    Arguments:
        
        -model      : The trained model
                      
        -model_name : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                      
        -stage_no   : The stage of training. This pipeline is trained in two stages 1 and 2. 
                      The stage number is needed to save the architecture for individual stages
                      and have unique file names
    """

    with open(path_dict['model_path']+"stage{}/".format(stage_no)+"{}_model_summary_stage_{}.txt".format(model_name, stage_no), "w") as f:
        with redirect_stdout(f):
            model.summary()
        
def save_params(input_params, path_dict):
    """
    This block of code is used to save the hyperparameter
    values into a csv file in the evaluation folder.
    
        Arguments:
        
        -input_params  :  This parameter will contain all the information that the user will
                          input through the terminal
    """

    with open(path_dict['sim_path'] + '/hyperparameters.csv', 'w') as f:
        f.write("%s,%s\n"%("hyperparameter","value\n"))
        for key in input_params.keys():
            f.write("%s,%s\n"%(key,input_params[key]))

def init_sizes():
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
    return size_dict

def load_models(model_name, path_dict):
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

    if(model_name=="vgg16"):
        base_model = vgg16.VGG16(weights=None, include_top=False)
        base_model.load_weights(path_dict["weights_path"]+"vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="inceptionv3"):
        base_model = inception_v3.InceptionV3(weights=None, include_top=False)
        base_model.load_weights(path_dict["weights_path"]+"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="resnet50"):
        base_model = resnet.ResNet50(weights=None, include_top=False)
        base_model.load_weights(path_dict["weights_path"]+"resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="inception_resnet"):
        base_model = inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False)
        base_model.load_weights(path_dict["weights_path"]+"inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="nasnet"):
        base_model = nasnet.NASNetLarge(weights=None, include_top=False)
        base_model.load_weights(path_dict["weights_path"]+"NASNet-large-no-top.h5")
    elif(model_name=="xception"):
        base_model = xception.Xception(weights=None, include_top=False)
        base_model.load_weights(path_dict["weights_path"]+"xception_weights_tf_dim_ordering_tf_kernels.h5")
    return base_model

#Define custom callbacks
def callbacks_list(input_params, stage_n, path_dict):
    """
    This function is used to define custom callbacks. Any new callbacks
    that are to be added to the model must be defined in this function
    and returned as a list of callbacks.
    
    Arguments:
        
        -input_params  :  This parameter will contain all the information that the user will
                          input through the terminal
                      
        -stage_n       :  The stage of training. This pipeline is trained in two stages 1 and 2. 
                          The stage number is needed to save the architecture for individual stages
                          and have unique file names
    """
    
    
    filepath = path_dict['model_path']+"stage{}/".format(stage_n)+"{}_weights_stage_{}.hdf5".format(input_params['model_name'], stage_n)
    
    checkpoint = ModelCheckpoint(filepath,
                                 monitor=input_params['monitor'],
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    
        
    reduce_learning_rate = ReduceLROnPlateau(monitor=input_params['monitor'],
                                             factor = 0.05,
                                             patience = 8)
    
    early_stop = EarlyStopping(monitor=input_params['monitor'], 
                               patience = 15)
    history = History()
    list_ = [checkpoint, reduce_learning_rate, history, early_stop]
    return list_

def no_of_classes(path_dict):
    """
    This function will be determine the number of classes that
    the model needs to train on. This function will determine
    the number of classes automatically without the user having
    to input the number of classes manually.
    """
    
    df = pd.read_msgpack(path_dict['df_path']+"train.msgpack")
    classes = df["class_label"].nunique()
    
    return classes

def train_stage1(input_params, path_dict):
    """
    In this stage, we will freeze all the convolution blocks and train
    only the newly added dense layers. We will add a global spatial average
    pooling layer, we will add fully connected dense layers on the output
    of the base models. We will freeze the convolution base and train only
    the top layers. We will set all the convolution layers to false, the model
    should be compiled when all the convolution layers are set to false.
    
    Arguments:
        
        -input_params  :  This parameter will contain all the information that the user will
                          input through the terminal
    """
    
    print("\nTraining the model by freezing the convolution block and tuning the top layers...")
    st = dt.now()
    base_model = load_models(input_params['model_name'], path_dict)
    #Adding a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    #Adding a fully-connected dense layer
    x = Dense(input_params['dense_neurons'], activation='relu', kernel_initializer='he_normal')(x)

    #Adding a final dense output final layer
    n = no_of_classes(path_dict)
    predictions = Dense(n, activation='softmax', kernel_initializer='glorot_uniform')(x)

    #Define the model
    model_stg1 = Model(inputs=base_model.input, outputs=predictions)

    #Here we will freeze the convolution base and train only the top layers
    #We will set all the convolution layers to false, the model should be
    #compiled when all the convolution layers are set to false
    for layer in base_model.layers:
        layer.trainable = False

    #Compiling the model
    model_stg1.compile(optimizer=optimizers.Adam(lr=input_params['stage1_lr']),
                       loss='categorical_crossentropy',
                       metrics=[input_params['metric']])

    #Normalize the images
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    df_train = load_data("train", path_dict)
    df_val = load_data("val", path_dict)

    size_dict=init_sizes()

    train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                        directory=path_dict['source'],
                                                        target_size=size_dict[input_params['model_name']],
                                                        x_col="filenames",
                                                        y_col="class_label",
                                                        batch_size=input_params['batch_size'],
                                                        class_mode='categorical',
                                                        color_mode='rgb',
                                                        shuffle=True)

    val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                    directory=path_dict['source'],
                                                    target_size=size_dict[input_params['model_name']],
                                                    x_col="filenames",
                                                    y_col="class_label",
                                                    batch_size=input_params['batch_size'],
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle=True)

    nb_train_samples = len(train_generator.classes)
    nb_val_samples = len(val_generator.classes)
    
    history=model_stg1.fit_generator(generator=train_generator,
                                     steps_per_epoch=nb_train_samples // input_params['batch_size'],
                                     epochs=input_params['epochs1'],
                                     validation_data=val_generator,
                                     validation_steps=nb_val_samples // input_params['batch_size'],
                                     callbacks=callbacks_list(input_params,1, path_dict)) #1 for stage 1


    hist_df = pd.DataFrame(history.history)
    hist_csv_file = path_dict['model_path'] + "stage{}/".format(1) + "{}_history_stage_{}.csv".format(input_params['model_name'],1)
    with open(hist_csv_file, mode='w') as file:
        hist_df.to_csv(file, index=None)

    model_stg1.load_weights(path_dict['model_path'] + "stage{}/".format(1) + "{}_weights_stage_{}.hdf5".format(input_params['model_name'],1))
    model_stg1.save(path_dict['model_path'] + "stage{}/".format(1) + "{}_model_stage_{}.h5".format(input_params['model_name'],1))

    save_summary(model_stg1, input_params['model_name'], 1, path_dict)
    plot_layer_arch(model_stg1, input_params['model_name'], 1, path_dict)

    stage1_params=dict()
    stage1_params['train_generator']=train_generator
    stage1_params['val_generator']=val_generator
    stage1_params['nb_train_samples']=nb_train_samples
    stage1_params['nb_val_samples']=nb_val_samples

    print("\nTime taken to train the model in stage 1: ",dt.now()-st)

    print("\nStarting model evaluation for stage 1..")
    evaluation.predict_on_test(input_params['model_name'],init_sizes(),1, path_dict)

    return model_stg1, stage1_params, size_dict

def train_stage2(input_params, stage1_params, model_stg2, path_dict):
    """
    At this point, the top layers are well trained and we can start fine-tuning
    convolutional layers of the pre-trained architecture. We will freeze the bottom
    x layers and train the remaining top layers. We will train the top N-x blocks and
    we will freeze the first x layers. A very low learning rate is used, so as to
    not wreck the convolution base with massive gradient updates. For training on
    stage 2, it's a good idea to double the number of epochs to train the model
    since the learning rate used for stage 2 is kept extremely low.

    Please refer to layer_inspection.py file for more info on how the layers are
    selected.
    
    Arguments:
        
        -input_params  :  This parameter will contain all the information that the user will
                          input through the terminal
                      
        -stage1_params :  This variable will contain all the parameters that are used in stage 1
                          so that it can be carried forward to stage 2 training phase without
                          re-defining them once again.
                          
        -model_stg2    :  This is nothing but the trained model in stage 1. This is passed to 
                          stage 2 for fine tuning the convolution block
                          
    """

    print("\nTraining the model by tuning the top convolution block along with the dense layers and freezing the rest...")
    st = dt.now()

    if(input_params['model_name']=='vgg16'):
        for layer in model_stg2.layers[:15]:
           layer.trainable = False
        for layer in model_stg2.layers[15:]:
           layer.trainable = True
    elif(input_params['model_name']=="inceptionv3"):
        for layer in model_stg2.layers[:249]:
           layer.trainable = False
        for layer in model_stg2.layers[249:]:
           layer.trainable = True
    elif(input_params['model_name']=="resnet50"):
        for layer in model_stg2.layers[:143]:
           layer.trainable = False
        for layer in model_stg2.layers[143:]:
           layer.trainable = True
    elif(input_params['model_name']=="inception_resnet"):
        for layer in model_stg2.layers[:759]:
           layer.trainable = False
        for layer in model_stg2.layers[759:]:
           layer.trainable = True
    elif(input_params['model_name']=="nasnet"):
        for layer in model_stg2.layers[:714]:
           layer.trainable = False
        for layer in model_stg2.layers[714:]:
           layer.trainable = True
    elif(input_params['model_name']=="xception"):
        for layer in model_stg2.layers[:106]:
           layer.trainable = False
        for layer in model_stg2.layers[106:]:
           layer.trainable = True

    #Recompile the model, train the top 2 blocks
    model_stg2.compile(optimizer=optimizers.Adam(lr=input_params['stage2_lr']),
                       loss='categorical_crossentropy',
                       metrics=[input_params['metric']])

    #Re-train the model again (this time fine-tuning the top 2 inception blocks
    #alongside the top Dense layers
    history=model_stg2.fit_generator(generator=stage1_params['train_generator'],
                                     steps_per_epoch=stage1_params['nb_train_samples'] // input_params['batch_size'],
                                     epochs=input_params['epochs2'],
                                     validation_data=stage1_params['val_generator'],
                                     validation_steps=stage1_params['nb_val_samples'] // input_params['batch_size'],
                                     callbacks=callbacks_list(input_params,2,path_dict))

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = path_dict['model_path'] + "stage{}/".format(2) + "{}_history_stage_{}.csv".format(input_params['model_name'],2)
    with open(hist_csv_file, mode='w') as file:
        hist_df.to_csv(file, index=None)

    model_stg2.load_weights(path_dict['model_path'] + "stage{}/".format(2)+"{}_weights_stage_{}.hdf5".format(input_params['model_name'],2))
    model_stg2.save(path_dict['model_path'] + "stage{}/".format(2) + "{}_model_stage_{}.h5".format(input_params['model_name'],2))

    save_summary(model_stg2, input_params['model_name'], 2, path_dict)
    plot_layer_arch(model_stg2, input_params['model_name'], 2, path_dict)

    print("\nStarting model evaluation for stage 2..")
    evaluation.predict_on_test(input_params['model_name'],init_sizes(),2, path_dict)

    print("\nTime taken to train the model in stage 2: ",dt.now()-st)

def train(input_params, path_dict):
    """
    This is a generic function which will be used to make
    other function calls to start training the model at
    all the stages. The model will be trained on the second
    stage if and only yes the "finetune" parameter has been
    set to 'yes' or else, only the first stage will train.
    By default, both the stages will train, so you will have
    to explicitly set the value of 'finetune' to 'no'.
    
    Arguments:
        
        -input_params  :  This parameter will contain all the information that the user will
                          input through the terminal
    """
        
    st=dt.now()
    model_stg1, stage1_params, size_dict = train_stage1(input_params, path_dict)
    if(input_params['finetune']=='yes'):
        train_stage2(input_params, stage1_params, model_stg1, path_dict)

    print("\nAll models are trained sucessfully..")
    print("\nTime taken to train both the models : ",dt.now()-st)
    print("\nAll model attributes are saved in this path: ",path_dict['model_path'])
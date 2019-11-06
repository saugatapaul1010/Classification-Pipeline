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
import argparse
from datetime import datetime as dt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
import eval_pipeline

#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5 --no-check-certificate
#!wget https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5 --no-check-certificate       

df_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_df/"
model_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/models/"
weights_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/weights/"
source="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data/"
os.mkdir(model_path) if not os.path.isdir(model_path) else None

def load_data():
    """
    This function is used to load the training and
    validation image dataframes.
    """
    df_train=pd.read_csv(df_path+"train.csv")
    df_val=pd.read_csv(df_path+"val.csv")
    return df_train, df_val

def init_sizes():
    """
    This block of code is used to initialize the input sizes
    of the images for specific models. Because specific models
    are trained using input images of specific sizes.
    """
    size_dict = dict()
    size_dict["vgg16"] = (224, 224)
    size_dict["inceptionv3"] = (299, 299)
    size_dict["resnet50"] = (224, 224)
    size_dict["inception_resnet"] = (299, 299)
    size_dict["nasnet"] = (331, 331)
    size_dict["xception"] = (299, 299)
    return size_dict
    
def load_models(model_name):
    """
    Initialize the pre-trained model architecture and load the model weights.
    The downloaded weights contains only the convolution base. It does not 
    contain the top two dense layers. We will have to manually define the top
    two dense layers. The size_dict dictionary object will hold the input sizes
    for various models, which will be further used to train the respective models
    with the given input image dimensions.
    """
       
    if(model_name=="vgg16"):
        base_model = vgg16.VGG16(weights=None, include_top=False)
        base_model.load_weights(weights_path+"vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="inceptionv3"):
        base_model = inception_v3.InceptionV3(weights=None, include_top=False)
        base_model.load_weights(weights_path+"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="resnet50"):
        base_model = resnet.ResNet50(weights=None, include_top=False)
        base_model.load_weights(weights_path+"resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    elif(model_name=="inception_resnet"):
        base_model = inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False)
        base_model.load_weights(weights_path+"inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")        
    elif(model_name=="nasnet"):
        base_model = nasnet.NASNetLarge(weights=None, include_top=False)
        base_model.load_weights(weights_path+"NASNet-large-no-top.h5")   
    elif(model_name=="xception"):
        base_model = xception.Xception(weights=None, include_top=False)
        base_model.load_weights(weights_path+"inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")      
    return base_model

#Define custom callbacks
def callbacks_list(input_params, stage_n):
    """
    This function is used to define custom callbacks. Any new callbacks
    that are to be added to the model must be defined in this function
    and returned as a list of callbacks.
    """
    checkpoint = ModelCheckpoint(model_path+"{}_weights_stage_{}.hdf5".format(input_params['model_name'],stage_n), 
                                 monitor=input_params['monitor'], 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto',
                                 period=1)

    reduce_learning_rate = ReduceLROnPlateau(monitor=input_params['monitor'], patience=10)
    early_stop = EarlyStopping(monitor=input_params['monitor'], patience=10)
    history = History()
    list_ = [checkpoint, reduce_learning_rate, early_stop, history]
    return list_

def train_stage1(input_params):
    """
    In this stage, we will freeze all the convolution blocks and train
    only the newly added dense layers. We will add a global spatial average 
    pooling layer, we will add fully connected dense layers on the output
    of the base models. We will freeze the convolution base and train only 
    the top layers. We will set all the convolution layers to false, the model 
    should be compiled when all the convolution layers are set to false.
    """
    print("\nTraining the model by freezing the convolution block and tuning the top layers...")
    st = dt.now()
    base_model = load_models(input_params['model_name'])
    #Adding a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    #Adding a fully-connected dense layer
    x = Dense(input_params['dense_neurons'], activation='relu', kernel_initializer='he_normal')(x)

    #Adding a final dense output final layer
    predictions = Dense(input_params['classes'], activation='softmax', kernel_initializer='glorot_uniform')(x)

    #Define the model
    model_stg1 = Model(inputs=base_model.input, outputs=predictions)

    #Here we will freeze the convolution base and train only the top layers
    #We will set all the convolution layers to false, the model should be
    #compiled when all the convolution layers are set to false
    for layer in base_model.layers:
        layer.trainable = False

    #Compiling the model
    model_stg1.compile(optimizer=optimizers.RMSprop(lr=input_params['stage1_lr']),
                       loss='categorical_crossentropy', 
                       metrics=[input_params['metric']])
    
    #Normalize the images
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
                  
    df_train, df_val = load_data()
    
    size_dict=init_sizes()
    
    train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                        directory=source,
                                                        target_size=size_dict[input_params['model_name']],
                                                        x_col="filenames",
                                                        y_col="class_label",                                                    
                                                        batch_size=input_params['batch_size'],
                                                        class_mode='categorical')

    val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                    directory=source,
                                                    target_size=size_dict[input_params['model_name']],
                                                    x_col="filenames",
                                                    y_col="class_label",                                                    
                                                    batch_size=input_params['batch_size'],
                                                    class_mode='categorical')

    nb_train_samples = len(train_generator.classes)
    nb_val_samples = len(val_generator.classes)
    
    history=model_stg1.fit_generator(generator=train_generator,
                                     steps_per_epoch=nb_train_samples // input_params['batch_size'],
                                     epochs=input_params['epochs'],
                                     validation_data=val_generator,
                                     validation_steps=nb_val_samples // input_params['batch_size'],
                                     callbacks=callbacks_list(input_params,1)) #1 for stage 1
    
        
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = model_path+"{}_history_stage_{}.csv".format(input_params['model_name'],1)
    with open(hist_csv_file, mode='w') as file:
        hist_df.to_csv(file, index=None)
    
    model_stg1.load_weights(model_path+"{}_weights_stage_{}.hdf5".format(input_params['model_name'],1))
    model_stg1.save(model_path+"{}_model_stage_{}.h5".format(input_params['model_name'],1))
        
    stage1_params=dict()
    stage1_params['train_generator']=train_generator
    stage1_params['val_generator']=val_generator
    stage1_params['nb_train_samples']=nb_train_samples
    stage1_params['nb_val_samples']=nb_val_samples
    
    print("\nTime taken to train the model in stage 1: ",dt.now()-st)
    
    print("\nStarting model evaluation for stage 1..")
    eval_pipeline.predict_on_test(input_params['model_name'],init_sizes(),1)
        
    return model_stg1, stage1_params, size_dict

def train_stage2(input_params, stage1_params, model_stg2):
    """
    At this point, the top layers are well trained and we can start fine-tuning
    convolutional layers of the pre-trained architecture. We will freeze the bottom 
    N layers and train the remaining top layers. We will train the top N blocks and 
    we will freeze the first N layers. A very low learning rate is used, so as to 
    not wreck the convolution base with massive gradient updates
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
        for layer in model_stg2.layers[:742]:
           layer.trainable = False
        for layer in model_stg2.layers[742:]:
           layer.trainable = True
    elif(input_params['model_name']=="inception_resnet"):
        for layer in model_stg2.layers[:758]:
           layer.trainable = False
        for layer in model_stg2.layers[758:]:
           layer.trainable = True        
    elif(input_params['model_name']=="nasnet"):
        for layer in model_stg2.layers[:15]:
           layer.trainable = False
        for layer in model_stg2.layers[15:]:
           layer.trainable = True   
    elif(input_params['model_name']=="xception"):
        for layer in model_stg2.layers[:15]:
           layer.trainable = False
        for layer in model_stg2.layers[15:]:
           layer.trainable = True
           
    #Recompile the model, train the top 2 blocks
    model_stg2.compile(optimizer=optimizers.RMSprop(lr=input_params['stage2_lr']), 
                       loss='categorical_crossentropy', 
                       metrics=[input_params['metric']])

    #Re-train the model again (this time fine-tuning the top 2 inception blocks
    #alongside the top Dense layers
    history=model_stg2.fit_generator(generator=stage1_params['train_generator'],
                                     steps_per_epoch=stage1_params['nb_train_samples'] // input_params['batch_size'],
                                     epochs=input_params['epochs'],
                                     validation_data=stage1_params['val_generator'],
                                     validation_steps=stage1_params['nb_val_samples'] // input_params['batch_size'],
                                     callbacks=callbacks_list(input_params,2))
    
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = model_path+"{}_history_stage_{}.csv".format(input_params['model_name'],2)
    with open(hist_csv_file, mode='w') as file:
        hist_df.to_csv(file, index=None)
        
    model_stg2.load_weights(model_path+"{}_weights_stage_{}.hdf5".format(input_params['model_name'],2))
    model_stg2.save(model_path+"{}_model_stage_{}.h5".format(input_params['model_name'],2))
    
    print("\nTime taken to train the model in stage 2: ",dt.now()-st)
                  
def train(input_params):
    """
    This is a generic function which will be used to make
    other function calls to start training the model at
    all the stages.
    """
    st=dt.now()
    model_stg1, stage1_params, size_dict = train_stage1(input_params)
    if(input_params['finetune']=='yes'):
        train_stage2(input_params, stage1_params, model_stg1)
        print("\nStarting model evaluation for stage 2..")
        eval_pipeline.predict_on_test(input_params['model_name'],init_sizes(),2)
        
    print("\nAll models are trained sucessfully..")
    print("\nTime taken to train both the models : ",dt.now()-st)
    print("\nAll model attributes are saved in this path: ",model_path)
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this script will train 3 machine learning models using transfer learning')
    parser.add_argument('--model_name', type=str, default='vgg16', help='choose the type of model you want to train with')
    parser.add_argument('--dense_neurons', type=int, default=1024, help='eneter the number of neurons you want for the pre-final layer')
    parser.add_argument('--classes', type=int, default=3, help="the number of classes for which the model should be trained on")
    parser.add_argument('--batch_size', type=int, default=5, help="enter the number of batches for which the model should be trained on")
    parser.add_argument('--stage1_lr', type=float, default=0.001, help="enter the learning rate for stage 1 training")
    parser.add_argument('--stage2_lr', type=float, default=0.00001, help="enter the learning rate for stage 2 training")
    parser.add_argument('--monitor',type=str, default='val_accuracy', help="enter the metric you want to monitor")
    parser.add_argument('--metric',type=str, default='accuracy', help="enter the metric you want the model to optimize")
    parser.add_argument('--epochs',type=int, default=5, help="enter the number of epochs you want the model to train for")
    parser.add_argument('--finetune',type=str, default='yes', help="state 'yes' or 'no' to say whether or not you want to fine tune the convolution block")
    args = parser.parse_args()
        
    input_params=dict()
    input_params['model_name']=args.model_name
    input_params['dense_neurons']=args.dense_neurons
    input_params['classes']=args.classes
    input_params['batch_size']=args.batch_size
    input_params['stage1_lr']=args.stage1_lr
    input_params['stage2_lr']=args.stage2_lr
    input_params['monitor']=args.monitor
    input_params['metric']=args.metric
    input_params['epochs']=args.epochs   
    input_params['finetune']=args.finetune
    
    train(input_params)
    

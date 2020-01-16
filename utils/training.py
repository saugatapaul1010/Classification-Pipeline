#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 19:18:20 2019

@author: saugata paul
"""

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
from evaluation import EvalUtils
from keras.utils import plot_model
from contextlib import redirect_stdout
from time import time
from keras.callbacks.tensorboard_v1 import TensorBoard
from utils import Utility

#FORCE GPU USE
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5 --no-check-certificate

class TrainingUtils:
    
    def __init__(self, input_params, path_dict):
        self.input_params = input_params
        self.path_dict = path_dict
        
    def plot_layer_arch(self, model, stage_no):
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
        
        plot_model(model, 
                   to_file=self.path_dict['model_path']+"stage{}/".format(stage_no)+'{}_model_architecture_stage_{}.pdf'.format(self.input_params["model_name"],stage_no), 
                   show_shapes=True, 
                   show_layer_names=True)

    
    def save_summary(self, model, stage_no):
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
    
        with open(self.path_dict['model_path']+"stage{}/".format(stage_no)+"{}_model_summary_stage_{}.txt".format(self.input_params["model_name"], stage_no), "w") as f:
            with redirect_stdout(f):
                model.summary()
            
    def save_params(self):
        """
        This block of code is used to save the hyperparameter
        values into a csv file in the evaluation folder.
        
            Arguments:
            
            -input_params  :  This parameter will contain all the information that the user will
                              input through the terminal
        """
    
        with open(self.path_dict['sim_path'] + '/hyperparameters.csv', 'w') as f:
            f.write("%s,%s\n"%("hyperparameter","value\n"))
            for key in self.input_params.keys():
                f.write("%s,%s\n"%(key,self.input_params[key]))
    
    def callbacks_list(self, stage_no):
        """
        This function is used to define custom callbacks. Any new callbacks
        that are to be added to the model must be defined in this function
        and returned as a list of callbacks.
        
        Arguments:
            
            -input_params  :  This parameter will contain all the information that the user will
                              input through the terminal
                          
            -stage_no       :  The stage of training. This pipeline is trained in two stages 1 and 2. 
                              The stage number is needed to save the architecture for individual stages
                              and have unique file names
        """
        
        filepath = self.path_dict['model_path']+"stage{}/".format(stage_no)+"{}_weights_stage_{}.hdf5".format(self.input_params['model_name'], stage_no)
        
        checkpoint = ModelCheckpoint(filepath,
                                     monitor=self.input_params['monitor'],
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        
            
        reduce_learning_rate = ReduceLROnPlateau(monitor=self.input_params['monitor'],
                                                 factor = 0.1,
                                                 patience = 3)
        
        early_stop = EarlyStopping(monitor=self.input_params['monitor'], 
                                   patience = 5)
        history = History()
        
        #Custom callback to monitor both validation accuracy and loss
        
        """
        best_val_acc = 0
        best_val_loss = sys.float_info.max 
        
        def saveModel(epoch,logs):
            val_acc = logs['val_acc']
            val_loss = logs['val_loss']
        
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save(...)
            elif val_acc == best_val_acc:
                if val_loss < best_val_loss:
                    best_val_loss=val_loss
                    model.save(...)
        
        callbacks = [LambdaCallback(on_epoch_end=saveModel)]
        """
        
        tensorboard = TensorBoard(log_dir=self.path_dict['model_path']+"stage{}/".format(stage_no)+"logs/{}".format(time),
                                  histogram_freq=0, 
                                  batch_size=32, 
                                  write_graph=True, 
                                  write_grads=False, 
                                  write_images=False, 
                                  embeddings_freq=0, 
                                  embeddings_layer_names=None, 
                                  embeddings_metadata=None, 
                                  embeddings_data=None, 
                                  update_freq='epoch')
        
        #!tensorboard --logdir=/home/developer/Desktop/Saugata/Classification-Pipeline/simulations/SIM_01/models/stage1/logs/
        
        list_ = [checkpoint, reduce_learning_rate, history, early_stop, tensorboard]
        return list_
    
    
    def train_stage1(self):
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
        
        utils_obj = Utility(self.input_params, self.path_dict)
        
        #Put if statement here. If model_name != custom then run this block, or else. Do something else.
        
        base_model = utils_obj.load_imagenet_model()
        
        #Adding a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    
        #Adding a fully-connected dense layer
        x = Dense(self.input_params['dense_neurons'], activation='relu', kernel_initializer='he_normal')(x)
    
        #Adding a final dense output final layer
        n = utils_obj.no_of_classes()
        output_layer = Dense(n, activation='softmax', kernel_initializer='glorot_uniform')(x)
    
        #Define the model
        model_stg1 = Model(inputs=base_model.input, outputs=output_layer)
    
        #Here we will freeze the convolution base and train only the top layers
        #We will set all the convolution layers to false, the model should be
        #compiled when all the convolution layers are set to false
        for layer in base_model.layers:
            layer.trainable = False
            
        #Compiling the model
        model_stg1.compile(optimizer=optimizers.Adam(lr=self.input_params['stage1_lr']),
                           loss='categorical_crossentropy',
                           metrics=[self.input_params['metric']])
    
        #Normalize the images
        train_datagen = ImageDataGenerator(preprocessing_function=utils_obj.init_preprocess_func())
        val_datagen = ImageDataGenerator(preprocessing_function=utils_obj.init_preprocess_func())
    
        df_train = utils_obj.load_data("train")
        df_val = utils_obj.load_data("val")
    
        train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                            directory=self.path_dict['source'],
                                                            target_size=utils_obj.init_sizes(),
                                                            x_col="filenames",
                                                            y_col="class_label",
                                                            batch_size=self.input_params['batch_size'],
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            shuffle=True)
    
        val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                        directory=self.path_dict['source'],
                                                        target_size=utils_obj.init_sizes(),
                                                        x_col="filenames",
                                                        y_col="class_label",
                                                        batch_size=self.input_params['batch_size'],
                                                        class_mode='categorical',
                                                        color_mode='rgb',
                                                        shuffle=True)
    
        nb_train_samples = len(train_generator.classes)
        nb_val_samples = len(val_generator.classes)
        
        history=model_stg1.fit_generator(generator=train_generator,
                                         steps_per_epoch=nb_train_samples // self.input_params['batch_size'],
                                         epochs=self.input_params['epochs1'],
                                         validation_data=val_generator,
                                         validation_steps=nb_val_samples // self.input_params['batch_size'],
                                         callbacks=TrainingUtils.callbacks_list(self, 1)) #1 for stage 1
    
    
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = self.path_dict['model_path'] + "stage{}/".format(1) + "{}_history_stage_{}.csv".format(self.input_params['model_name'], 1)
        with open(hist_csv_file, mode='w') as file:
            hist_df.to_csv(file, index=None)
    
        model_stg1.load_weights(self.path_dict['model_path'] + "stage{}/".format(1) + "{}_weights_stage_{}.hdf5".format(self.input_params['model_name'], 1))
        model_stg1.save(self.path_dict['model_path'] + "stage{}/".format(1) + "{}_model_stage_{}.h5".format(self.input_params['model_name'], 1))

        TrainingUtils.save_summary(self, model_stg1, 1)
        TrainingUtils.plot_layer_arch(self, model_stg1, 1)
    
        stage1_params=dict()
        stage1_params['train_generator']=train_generator
        stage1_params['val_generator']=val_generator
        stage1_params['nb_train_samples']=nb_train_samples
        stage1_params['nb_val_samples']=nb_val_samples
    
        print("\nTime taken to train the model in stage 1: ",dt.now()-st)
        
        #Start model evaluation for Stage 1
        eval_utils = EvalUtils(self.input_params, self.path_dict, 1)
        eval_utils.predict_on_test()
    
        return model_stg1, stage1_params
    
    def train_stage2(self):
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
        
        model_stg2 = self.model_stg1
    
        if(self.input_params['model_name']=='vgg16'):
            for layer in model_stg2.layers[:15]:
               layer.trainable = False
            for layer in model_stg2.layers[15:]:
               layer.trainable = True
        elif(self.input_params['model_name']=="inceptionv3"):
            for layer in model_stg2.layers[:249]:
               layer.trainable = False
            for layer in model_stg2.layers[249:]:
               layer.trainable = True
        elif(self.input_params['model_name']=="resnet50"):
            for layer in model_stg2.layers[:143]:
               layer.trainable = False
            for layer in model_stg2.layers[143:]:
               layer.trainable = True
        elif(self.input_params['model_name']=="inception_resnet"):
            for layer in model_stg2.layers[:759]:
               layer.trainable = False
            for layer in model_stg2.layers[759:]:
               layer.trainable = True
        elif(self.input_params['model_name']=="nasnet"):
            for layer in model_stg2.layers[:714]:
               layer.trainable = False
            for layer in model_stg2.layers[714:]:
               layer.trainable = True
        elif(self.input_params['model_name']=="xception"):
            for layer in model_stg2.layers[:106]:
               layer.trainable = False
            for layer in model_stg2.layers[106:]:
               layer.trainable = True
    
        #Recompile the model, train the top 2 blocks
        model_stg2.compile(optimizer=optimizers.Adam(lr=self.input_params['stage2_lr']),
                           loss='categorical_crossentropy',
                           metrics=[self.input_params['metric']])
    
        #Re-train the model again (this time fine-tuning the top 2 inception blocks
        #alongside the top Dense layers
        history=model_stg2.fit_generator(generator=self.stage1_params['train_generator'],
                                         steps_per_epoch=self.stage1_params['nb_train_samples'] // self.input_params['batch_size'],
                                         epochs=self.input_params['epochs2'],
                                         validation_data=self.stage1_params['val_generator'],
                                         validation_steps=self.stage1_params['nb_val_samples'] // self.input_params['batch_size'],
                                         callbacks=TrainingUtils.callbacks_list(self, 2))
    
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = self.path_dict['model_path'] + "stage{}/".format(2) + "{}_history_stage_{}.csv".format(self.input_params['model_name'], 2)
        with open(hist_csv_file, mode='w') as file:
            hist_df.to_csv(file, index=None)
    
        model_stg2.load_weights(self.path_dict['model_path'] + "stage{}/".format(2)+"{}_weights_stage_{}.hdf5".format(self.input_params['model_name'], 2))
        model_stg2.save(self.path_dict['model_path'] + "stage{}/".format(2) + "{}_model_stage_{}.h5".format(self.input_params['model_name'], 2))
    
        TrainingUtils.save_summary(self, model_stg2, 2)
        TrainingUtils.plot_layer_arch(self, model_stg2, 2)
            
        #Start model evaluation for Stage 2
        eval_utils = EvalUtils(self.input_params, self.path_dict, 2)
        eval_utils.predict_on_test()
    
        print("\nTime taken to train the model in stage 2: ",dt.now()-st)
    
    def train(self):
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
        self.model_stg1, self.stage1_params = TrainingUtils.train_stage1(self)
        
        if(self.input_params['finetune']=='yes'):
            TrainingUtils.train_stage2(self)
        else:
            pass
    
        print("\nAll models are trained sucessfully..")
        print("\nTime taken to train both the models : ",dt.now()-st)
        print("\nAll model attributes are saved in this path: ",self.path_dict['model_path'])
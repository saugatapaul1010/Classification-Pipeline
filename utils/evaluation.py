#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:10:21 2019

@author: saugata paul
"""

import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import pycm
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from keras.applications.resnet import preprocess_input as preprocess_input_resnet50
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from utils import Utility



#FORCE GOU USE
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class EvalUtils:
    
    def __init__(self, input_params, path_dict, stage_no):
        self.input_params = input_params
        self.path_dict = path_dict
        self.stage_no = stage_no

    #Plot loss vs epochs
    def plt_epoch_error(self, history_df):
        """
        This function is used to plot the loss vs epoch for the
        trained models using the History callback and save the
        diagrams in the evaluation folders.
    
        Arguments:                    
    
            -model_name : Name of the model, for example - vgg16, inception_v3, resnet50 etc
            
            -stage_no   : The stage of training for which the evaluation has tp be done. This pipeline 
                          is trained in two stages 1 and 2. The stage number is needed to save the 
                          architecture for individual stages and have unique file names. 
                          
            -history_df : For each training phase, 'history_df' dataframe will be created, which will
                          contain the loss vs epoch details for each phase. This dataframe is used to
                          plot the loss vs epoch curve.    
        
        """
        
        epochs = len(history_df.loss.values)
    
        plt.figure(figsize=(12,6))
        plt.grid()
        plt.plot(list(range(epochs)), history_df.val_loss.values, 'blue',
                 label="Validation Loss", linewidth=5)
        plt.plot(list(range(epochs)), history_df.loss.values, 'red', label="Training Loss", linewidth=5)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("train vs validation loss for stage {}.".format(self.stage_no))
        plt.legend()
    
        plt.savefig(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_history_stage_{}.png'.format(self.input_params['model_name'],self.stage_no))
        print("\nFile saved at this location: "+self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_history_stage_{}.png'.format(self.input_params['model_name'],self.stage_no))
    
    def get_metrics(self, y_true, y_pred):
        """
        This function is used to get only the list of important metrics
        and save them as a csv file in the evaluation folder. There will
        be a seperate function which will list down all the important
        class wise metrics. However, this function will contain only the
        most important metrics for the classification problem that we are
        trying to solve at hand.
        
        Arguments:                    
    
            -y_true : Ground truths
            
            -y_pred : Predicted labels
        """
    
        scores = dict()
        scores['acc_score'] = metrics.accuracy_score(y_true, y_pred)
        scores['f1_score'] = metrics.f1_score(y_true, y_pred, average='macro')
        scores['precision'] = metrics.precision_score(y_true, y_pred, average='macro')
        scores['recall'] = metrics.recall_score(y_true, y_pred, average='macro')
    
        df_metrics = pd.DataFrame()
        df_metrics["metrics"]=list(scores.keys())
        df_metrics["values"]=list(scores.values())
        print('Metrics computed and saved..')
    
        return df_metrics
    
    def get_complete_report(self, y_true, y_pred, class_indices):
        """
        This is a separate function written to calculate every possible
        classification metric value that different classification problems
        might need. This function will be used to get a report of all the
        classification metrics, as well the class wise statistics for all the
        classes and export it to a HTML file saved at the evaluation path.
    
        References to the library: https://www.pycm.ir/doc/index.html#Cite
    
          @article{Haghighi2018,
          doi = {10.21105/joss.00729},
          url = {https://doi.org/10.21105/joss.00729},
          year  = {2018},
          month = {may},
          publisher = {The Open Journal},
          volume = {3},
          number = {25},
          pages = {729},
          author = {Sepand Haghighi and Masoomeh Jasemi and Shaahin Hessabi and Alireza Zolanvari},
          title = {{PyCM}: Multiclass confusion matrix library in Python},
          journal = {Journal of Open Source Software}
          }
    
        Arguments:                    
    
            -y_true     : Ground truths
            
            -y_pred     : Predicted labels
            
            -model_name : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                
            -stage_no   : The stage of training for which the evaluation has tp be done. This pipeline 
                          is trained in two stages 1 and 2. The stage number is needed to save the 
                          architecture for individual stages and have unique file names.
                          
            -class_indices : This contains information about the mapping of the class labels to integers.
        """
    
        label_indices = dict()
        for (k,v) in class_indices.items():
            label_indices[v]=k
    
        y_true_label = list(y_true)
        y_pred_label = list(y_pred)
    
        for idx, item in enumerate(y_true_label):
            y_true_label[idx] = label_indices[item]
    
        for idx, item in enumerate(y_pred_label):
            y_pred_label[idx] = label_indices[item]
    
        cm = pycm.ConfusionMatrix(y_true_label, y_pred_label)
        cm.save_html(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_detailed_metrics_analysis_stage_{}'.format(self.input_params["model_name"],self.stage_no))
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """
        Based on the model name and the stage number, this function will be used
        to plot the confusion matrix, recall matrix and precision matrix and save
        them as image files in the evaluation folder.
        
        Arguments:                    
    
            -test_y         : Ground truths
            
            -predict_y      : Predicted labels
            
            -label          : Class labels, class indices
                
            -model_name     : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                          
            -stage_no       : The training stage of the model.
        """
        
        CM = metrics.confusion_matrix(y_true, y_pred)
        print("Percentage of misclassified points : ",(len(y_true)-np.trace(CM))/len(y_true)*100)
    
        RM = (((CM.T)/(CM.sum(axis=1))).T)
        PM = (CM/CM.sum(axis=0))
    
        cmap=sns.light_palette("green")
        
        #Representing CM in heatmap format
        plt.figure(figsize=(20,8))
        sns.heatmap(CM, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.title('{}_cm_matrix_stage_{}'.format(self.input_params["model_name"],self.stage_no))
        plt.savefig(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_cm_matrix_stage_{}.png'.format(self.input_params["model_name"],self.stage_no))
    
        #Representing PM in heatmap format
        plt.figure(figsize=(20,8))
        sns.heatmap(PM, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.title('{}_recall_matrix_stage_{}'.format(self.input_params["model_name"],self.stage_no))
        plt.savefig(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_precision_matrix_stage_{}.png'.format(self.input_params["model_name"],self.stage_no))
    
        #Representing RM in heatmap format
        plt.figure(figsize=(20,8))
        sns.heatmap(RM, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.title('{}_precsion_matrix_stage_{}'.format(self.input_params["model_name"],self.stage_no))
        plt.savefig(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_recall_matrix_stage_{}.png'.format(self.input_params["model_name"],self.stage_no))
    
    #self = input params, stage, path dict
    def predict_on_test(self):
        """
        This function will load the test dataset, pre-process the test
        images and check the performance of the trained models on unseen
        data. This will also save the confusion matrix and classification
        report as csv file in seperate dataframes for each model and for
        each stage, in the evaluation directory.
        
        Arguments:                    
            
            -size_dict    : Contains information about the image input image sizes for each of the models
                
            -model_name   : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                          
            -stage_no     : The training stage of the model. You will have a choice to select the number
                            of training stages. In stage 1, we only fine tune the top 2 dense layers by
                            freezing the convolution base. In stage 2, we will re adjust the weights trained
                            in stage 1 by training the top convolution layers, by freezing the dense layers.
        """
        
        #Create an utility class object to access the class methods
        utils_obj = Utility(self.input_params, self.path_dict)
        
        df_test = utils_obj.load_data("test")
        
        test_datagen = ImageDataGenerator(preprocessing_function=utils_obj.init_preprocess_func())
    
        test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
                                                          directory=self.path_dict['source'],
                                                          target_size=utils_obj.init_sizes(),
                                                          x_col="filenames",
                                                          y_col="class_label",
                                                          batch_size=1,
                                                          class_mode='categorical',
                                                          color_mode='rgb',
                                                          shuffle=False)
    
        nb_test_samples = len(test_generator.classes)
    
        model = utils_obj.get_models(self.stage_no)
        class_indices=test_generator.class_indices
    
        def label_class(cat_name):
            return(class_indices[cat_name])
    
        df_test['true']=df_test['class_label'].apply(lambda x: label_class(str(x)))
        y_true=df_test['true'].values
    
        #Predictions (Probability Scores and Class labels)
        y_pred_proba = model.predict_generator(test_generator, nb_test_samples // 1)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
        df_test['predicted'] = y_pred
        df_test.to_csv(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_predictions_stage_{}.csv'.format(self.input_params['model_name'],self.stage_no))
        dictionary = dict(zip(df_test.true.values, df_test.class_label.values))
    
        #Confusion Matrixs
        cm=metrics.confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm).transpose()
        df_cm=df_cm.rename(mapper=dict, index=dictionary, columns=dictionary, copy=True, inplace=False)
        df_cm.to_csv(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_cm_stage_{}.csv'.format(self.input_params['model_name'],self.stage_no))
        print('Confusion matrix prepared and saved..')
    
        #Classification Report
        report=metrics.classification_report(y_true,
                                             y_pred,
                                             target_names=list(class_indices.keys()),
                                             output_dict=True)
    
        df_rep = pd.DataFrame(report).transpose()
        df_rep.to_csv(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_class_report_stage_{}.csv'.format(self.input_params['model_name'],self.stage_no))
        print('Classification report prepared and saved..')
        
        EvalUtils.plot_confusion_matrix(self, y_true, y_pred, list(test_generator.class_indices.keys()))
    
        #General Metrics
        df_metrics = EvalUtils.get_metrics(self, y_true, y_pred)
        df_metrics.to_csv(self.path_dict["eval_path"]+"stage{}/".format(self.stage_no)+'{}_metrics_stage_{}.csv'.format(self.input_params['model_name'],self.stage_no))
    
        history_df=pd.read_csv(self.path_dict["model_path"]+"stage{}/".format(self.stage_no)+"{}_history_stage_{}.csv".format(self.input_params['model_name'], self.stage_no))
    
        #Get the train vs validation loss for all epochs
        EvalUtils.plt_epoch_error(self, history_df)
    
        #Generate a complete report and save it as an HTML file in the evaluation folder location
        EvalUtils.get_complete_report(self, y_true, y_pred, class_indices)
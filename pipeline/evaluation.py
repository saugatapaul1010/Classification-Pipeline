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
#from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
import argparse
import pycm
from PIL import Image
from keras.applications.vgg16 import preprocess_input

#FORCE GOU USE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

root_path = "/home/developer/deep_learning/workspace/Saugata_Paul/Classification_Pipeline/"

df_path = root_path + "data_df/"
model_path = root_path + "models/"
weights_path = root_path + "weights/"
source = root_path + "data/"
eval_path = root_path + "evaluation/"

os.mkdir(eval_path) if not os.path.isdir(eval_path) else None

def load_data(set_name):
    """
    This function is used to load the training, 
    validation as well as the test data.
    
    3 datasets are present => train.csv, val.csv, test.csv
    """

    df = pd.read_msgpack(df_path+"{}.msgpack".format(set_name))
    return df

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

def get_models(model_name, stage_no):
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
    model = load_model(model_path+"{}_model_stage_{}.h5".format(model_name, stage_no))
    return model

#Plot loss vs epochs
def plt_epoch_error(history_df, model_name, stage_no):
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
    plt.title("train vs validation loss for stage {}.".format(stage_no))
    plt.legend()

    plt.savefig(eval_path+'{}_history_stage_{}.png'.format(model_name,stage_no))
    print("\nFile saved at this location: "+eval_path+'{}_history_stage_{}.png'.format(model_name,stage_no))

def get_metrics(y_true, y_pred):
    """
    This function is used to get only the list of important metrics
    and save them as a CSV file in the evaluation folder. There will
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

def get_complete_report(y_true, y_pred, model_name, stage_no, class_indices):
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
    cm.save_html(eval_path+'{}_detailed_metrics_analysis_stage_{}'.format(model_name,stage_no))

def plot_confusion_matrix(test_y, predict_y, test_generator, model_name, stage_no):
    """
    Based on the model name and the stage number, this function will be used
    to plot the confusion matrix, recall matrix and precision matrix and save
    them as image files in the evaluation folder.
    
    Arguments:                    

        -test_y         : Ground truths
        
        -predict_y      : Predicted labels
        
        -test_generator : The test generator object is needed to get the class label indices
            
        -model_name     : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                      
        -stage_no       : The training stage of the model.
    """
    
    CM = metrics.confusion_matrix(test_y, predict_y)
    print("Percentage of misclassified points ",(len(test_y)-np.trace(CM))/len(test_y)*100)

    RM = (((CM.T)/(CM.sum(axis=1))).T)
    PM = (CM/CM.sum(axis=0))

    labels = list(test_generator.class_indices.keys())
    cmap=sns.light_palette("green")
    
    #Representing CM in heatmap format
    plt.figure(figsize=(20,8))
    sns.heatmap(CM, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title('{}_cm_matrix_stage_{}'.format(model_name,stage_no))
    plt.savefig(eval_path+'{}_cm_matrix_stage_{}.png'.format(model_name,stage_no))

    #Representing PM in heatmap format
    plt.figure(figsize=(20,8))
    sns.heatmap(PM, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title('{}_recall_matrix_stage_{}'.format(model_name,stage_no))
    plt.savefig(eval_path+'{}_precision_matrix_stage_{}.png'.format(model_name,stage_no))

    #Representing RM in heatmap format
    plt.figure(figsize=(20,8))
    sns.heatmap(RM, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title('{}_precsion_matrix_stage_{}'.format(model_name,stage_no))
    plt.savefig(eval_path+'{}_recall_matrix_stage_{}.png'.format(model_name,stage_no))


def predict_on_test(model_name, size_dict, stage_no):
    """
    This function will load the test dataset, pre-process the test
    images and check the performance of the trained models on unseen
    data. This will also save the confusion matrix and classification
    report as CSV file in seperate dataframes for each model and for
    each stage, in the evaluation directory.
    
    Arguments:                    
        
        -size_dict    : Contains information about the image input image sizes for each of the models
            
        -model_name   : Name of the model, for example - vgg16, inception_v3, resnet50 etc
                      
        -stage_no     : The training stage of the model. You will have a choice to select the number
                        of training stages. In stage 1, we only fine tune the top 2 dense layers by
                        freezing the convolution base. In stage 2, we will re adjust the weights trained
                        in stage 1 by training the top convolution layers, by freezing the dense layers.
    """

    df_test=load_data("test")
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
                                                      directory=source,
                                                      target_size=size_dict[model_name],
                                                      x_col="filenames",
                                                      y_col="class_label",
                                                      batch_size=1,
                                                      class_mode='categorical',
                                                      color_mode='rgb',
                                                      shuffle=False)

    nb_test_samples = len(test_generator.classes)

    model = get_models(model_name, stage_no)
    class_indices=test_generator.class_indices

    def label_class(cat_name):
        return(class_indices[cat_name])

    df_test['true']=df_test['class_label'].apply(lambda x: label_class(str(x)))
    y_true=df_test['true'].values

    #Confusion Matrix
    y_pred_proba = model.predict_generator(test_generator, nb_test_samples // 1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    #y_pred=  (y_pred_proba > 0.5)*1
    df_test['predicted']=y_pred

    df_test.to_csv(eval_path+'{}_predictions_stage_{}.csv'.format(model_name,stage_no))

    dictionary = dict(zip(df_test.true.values, df_test.class_label.values))

    cm=metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm).transpose()
    df_cm=df_cm.rename(mapper=dict, index=dictionary, columns=dictionary, copy=True, inplace=False)
    df_cm.to_csv(eval_path+'{}_cm_stage_{}.csv'.format(model_name,stage_no))
    print('Confusion matrix prepared and saved..')


    #Classification Report
    report=metrics.classification_report(y_true,
                                         y_pred,
                                         target_names=list(class_indices.keys()),
                                         output_dict=True)

    df_rep = pd.DataFrame(report).transpose()
    df_rep.to_csv(eval_path+'{}_class_report_stage_{}.csv'.format(model_name,stage_no))
    print('Classification report prepared and saved..')

    plot_confusion_matrix(y_true, y_pred, test_generator, model_name,stage_no)

    #General Metrics
    df_metrics = get_metrics(y_true, y_pred)
    df_metrics.to_csv(eval_path+'{}_metrics_stage_{}.csv'.format(model_name,stage_no))

    history_df=pd.read_csv(model_path+"{}_history_stage_{}.csv".format(model_name, stage_no))

    #Get the train vs validation loss for all epochs
    plt_epoch_error(history_df,model_name,stage_no)

    #Generate a complete report and save it as an HTML file in the evaluation folder location
    get_complete_report(y_true, y_pred, model_name, stage_no, class_indices)

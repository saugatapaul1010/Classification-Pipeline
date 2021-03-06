#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:27:02 2019

@author: saugata paul
"""

#Imports for pre-processing of images and other utlity libraries
import os
from datetime import datetime as dt
import pandas as pd
import argparse

root_path = "/home/developer/Desktop/Classification_Pipeline_Classes/"

df_path = root_path + "data"
destination = root_path + "data_df/"
os.mkdir(destination) if not os.path.isdir(destination) else None

def prepare_data(val_split, test_split, source):
    """
    This function is used to prepare the dataset in msgpack format. This function
    will take in a series of folders containing images and create 3 separate 
    msgpack files for train, validation and test data. The resultant dataset will
    contain the full path of the image as well as the corresponding class label.
    
    Arguments:
        
        -val_split  : Fraction of the training data that will be considered for 
                      validation
                      
        -test_split : Fraction of the original data that will be considered for
                      evaluating the model performance on unseen data
                      
        -source     : The full path of the data directory. This directory should 
                      contain the folders arranged according to the class labels 
                      needed for classification    
    """
    
    st = dt.now()

    delim="/"

    fol_names = os.listdir(source+delim)

    data_dict = dict()
    for i in range(0,len(fol_names)):
        fol_path = os.path.abspath(source) + delim + fol_names[i]
        fil_list = os.listdir(fol_path)
        for j in range(0,len(fil_list)):
            fil_path = fol_path + delim + fil_list[j]
            data_dict[fil_path] = fol_names[i]

    df=pd.DataFrame(list(data_dict.items()), columns=['filename','class_label'])
    df.to_msgpack(destination+"dataframe.msgpack")

    df=pd.read_msgpack(destination+"dataframe.msgpack")

    from sklearn.model_selection import train_test_split
    X, y = df['filename'].values, df['class_label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=val_split, random_state=42)

    df_val=pd.DataFrame(columns=["filenames","class_label"])
    df_val["filenames"] = X_val
    df_val["class_label"] = y_val

    df_test=pd.DataFrame(columns=["filenames","class_label"])
    df_test["filenames"] = X_test
    df_test["class_label"] = y_test

    df_train=pd.DataFrame(columns=["filenames","class_label"])
    df_train["filenames"] = X_train
    df_train["class_label"] = y_train

    df_train.to_msgpack(destination+"train.msgpack")
    df_val.to_msgpack(destination+"val.msgpack")
    df_test.to_msgpack(destination+"test.msgpack")

    print("Time taken to prepare the whole dataset: ",dt.now()-st)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare the original dataset into train, validation and test sets')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of train data to be used as validation data')
    parser.add_argument('--test_split', type=float, default=0.2, help='fraction of original data to used as test data')
    parser.add_argument('--source', type=str, default=df_path, help="source directory of the input images")
    args = parser.parse_args()

    st = dt.now()
    prepare_data(args.val_split, args.test_split, args.source)
    print("Train, Test and Validation dataset prepared successfully..")
    print("Time taken to prepare the dataset: ",dt.now()-st)

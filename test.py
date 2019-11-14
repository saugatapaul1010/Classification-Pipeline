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
from keras.utils import plot_model


#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5 --no-check-certificate
#!wget https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate
#!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-check-certificate



df_path="C:\\Users\\206255\\Desktop\\Saugata Paul\\Classification-pipeline-for-transfer-learning\\data_df\\"
model_path="C:\\Users\\206255\\Desktop\\Saugata Paul\\Classification-pipeline-for-transfer-learning\\models\\"
weights_path="C:\\Users\\206255\\Desktop\\Saugata Paul\\Classification-pipeline-for-transfer-learning\\weights\\"
source="C:\\Users\\206255\\Desktop\\Saugata Paul\\Classification-pipeline-for-transfer-learning\\data\\"

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


df_train, df_val = load_data()


















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

df_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data"
#df_path="C:\\Users\\206255\\Desktop\\Saugata Paul\\Classification-pipeline-for-transfer-learning\\data\\"


st = dt.now()

delim="/"

destination = "data_df/"
os.mkdir(destination) if not os.path.isdir(destination) else None

fol_names = os.listdir(source+delim)

data_dict = dict()
for i in range(0,len(fol_names)):
    fol_path = os.path.abspath(source) + delim + fol_names[i]
    fil_list = os.listdir(fol_path)
    for j in range(0,len(fil_list)):
        fil_path = fol_path + delim + fil_list[j]
        data_dict[fil_path] = fol_names[i]

df=pd.DataFrame(list(data_dict.items()), columns=['filename','class_label'])
df.to_csv(destination+"dataframe.csv", index=None)

df=pd.read_csv(destination+"dataframe.csv")

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

df_train.to_csv(destination+"train.csv", index=None)
df_val.to_csv(destination+"val.csv", index=None)
df_test.to_csv(destination+"test.csv", index=None)

print("Time taken to prepare the whole dataset: ",dt.now()-st)
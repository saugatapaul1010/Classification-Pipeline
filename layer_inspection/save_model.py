#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:21:45 2019

@author: saugata paul
"""

from keras.models import load_model

eval_path="/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/models/"

model=load_model(eval_path+"vgg16_model_stage_1.h5")

def save_summary(model, model_name, stage_no):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    
    with open(eval_path+"{}_model_summary_stage_{}.txt".format(model_name, stage_no), "w") as text_file:
        print(short_model_summary, file=text_file)

save_summary(model, "vgg16", 1)
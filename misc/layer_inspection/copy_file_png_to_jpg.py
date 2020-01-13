#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:16:32 2019

@author: saugata paul
"""

import glob
import os
import shutil
from tqdm import tqdm

src_png = "/home/developer/Desktop/Saugata/E-Crash/Classififcation_Pipeline/whole_data/"
src_jpg = "/media/developer/Storage1/FL_jpgs/"
dest_jpg = "/media/developer/Storage1/Classification_Data/CA/"

os.mkdir(dest_jpg) if not os.path.isdir(dest_jpg) else None
os.mkdir(dest_jpg+"Printed") if not os.path.isdir(dest_jpg+"Printed") else None
os.mkdir(dest_jpg+"Handwritten") if not os.path.isdir(dest_jpg+"Handwritten") else None


files = [file for file in glob.glob(src_png + "**/*.png", recursive=True)]
files_jpg = []
    
file_names = [name.replace(".png",".jpg").split("/")[-1] for name in files]
file_type = [name.replace(".png",".jpg").split("/")[-2] for name in files]

file_dict = dict(zip(file_names, file_type))

for k,v in tqdm(file_dict.items()):
    if(v=="printed"):
        try:
            shutil.copy(src_jpg+k, dest_jpg+"Printed")
        except FileNotFoundError:
            continue
    else:
        try:
            shutil.copy(src_jpg+k, dest_jpg+"Handwritten")
        except FileNotFoundError:
            continue
        
        
path = "/home/developer/Desktop/Saugata/E-Crash/outputs/500_handwritten_images/image_labels_CA_500.csv"

df = pd.read_csv(path)

df
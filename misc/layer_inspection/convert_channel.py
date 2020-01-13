#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:51:13 2019

@author: saugata paul
"""


import glob
from PIL import Image
import numpy as np
import os
import shutil
from tqdm import tqdm
import cv2

# Sachin Image (JPEG)
DIR_ = "/home/developer/Desktop/303156049_7.jpg"
image = Image.open(DIR_)
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]
#ch4=image[:,:,3] #This will give error

# 1-bit PNG Image
DIR_ = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_png1/handwritten/291854693_6.png"
image = Image.open(DIR_)
image.mode
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]

# 1-bit Image - Open CV
DIR_ = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_png1/handwritten/291854693_6.png"
image = cv2.imread(DIR_)
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]
ch4=image[:,:,3] #This will give error

# 1-bit to 8-bit conversion (1 channel)
DIR_ = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_png1/handwritten/291854693_6.png"
image = Image.open(DIR_)
image = image.convert("L")
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]

# 1-bit to 24-bit conversion (3 channels)
DIR_ = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_png1/handwritten/291854693_6.png"
image = Image.open(DIR_)
image = image.convert("RGB")
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]

# After image augmentation
DIR_ = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data/augmented/291854637_6_translate_0.png"
image = Image.open(DIR_)
image = image.convert("RGB")
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]

# Venky Images
from PIL import Image
import numpy as np

DIR = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/venky_samples/"
image_png_1_0 = Image.open(DIR + "Anaheim_Police_Department_303326073_0.png")
image_png_1_0_24 = image_png_1_0.convert("RGB")

image_jpg_1_1 = Image.open(DIR + "Anaheim_Police_Department_303326073_1.jpg")
image_jpg_1_1_24 = image_jpg_1_1.convert("RGB")

image_png_2_0 = Image.open(DIR + "Anaheim_Police_Department_303326077_0.png")
image_png_2_0_24 = image_png_2_0.convert("1")

image_jpg_2_1 = Image.open(DIR + "Anaheim_Police_Department_303326077_1.jpg")
image_jpg_2_1_24 = image_jpg_2_1.convert("RGB")

image_np = np.array(image_png_2_1_24)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]


from sys import getsizeof

# PNG 1 bit
png_1bit = getsizeof(np.array(image_png_1_0))

# PNG 24 bit
png_24bit = getsizeof(np.array(image_png_1_0_24))

# JPG 8 bit
jpg_8bit = getsizeof(np.array(image_jpg_1_1))

# JPG 24 bit
jpg_24bit = getsizeof(np.array(image_png_1_0_24))



name = "This is the inidian indian This is the inidian indian This is the inidian indian India!"
deep_getsizeof(name)



import sys
str1 = "one"
int_element=5
print("Memory size of '"+str1+"' = "+str(sys.getsizeof(str1))+ " bytes")
print("Memory size of '"+ str(int_element)+"' = "+str(sys.getsizeof(int_element))+ " bytes")




"""Image Conversion"""

dest = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_png/291853673_1.png"
source = "/home/    /Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/data_png/"

os.mkdir(dest) if not os.path.isdir(dest) else None
file_names = [file for file in glob.glob(source + "**/*.png", recursive=True)]

for path in tqdm(file_names):
    f_name = path.split("/")[-1]
    label = path.split("/")[-2]
    os.mkdir(dest+label) if not os.path.isdir(dest+label) else None
    
    image = Image.open(path)
    conv_image = image.convert("RGB")
    conv_image.save(dest+label+"/"+f_name.split(".")[0]+".png")
    
    
# Image after adding color in MS PAINT
DIR_ = "/home/developer/Desktop/Saugata/e-Crash/Classification-pipeline-for-transfer-learning/Analysis/red.png"
image = Image.open(DIR_)
image = image.convert("RGB")
image_np = np.array(image)
ch1=image_np[:,:,0]
ch2=image_np[:,:,1]
ch3=image_np[:,:,2]




image.__size_of__





var = 565656456665
print(getsizeof(var))














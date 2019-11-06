#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:02:46 2019

@author: developer
"""

'''
input_img_dir: Path of the input images that has to be augmented. (eg:'/home/developer/PROJECTS/Ecrash/image_classification/test_aug_inp_png/')
img_format: Extension of image. (eg:'*.png')
out_img_dir: Path of the output augmented images. (eg:'/home/developer/PROJECTS/Ecrash/image_classification/test_aug_out_png/')
no_of replications: Number of images need to be generated for each type of augmentation (eg:5)
rotate_min: Minimum value to rotate the image. (eg:-20)
rotate_max: Maximum value to rotate the image. (eg:20)
scale_min: Minimum value to scale the image. (eg:0.01)
scale_max: Maximum value to scale the image. (eg:0.10)
translate_min: Minimum value to translate the image. (eg:-0.2)
translate_max: Maximum value to translate the image. (eg:0.2)
'''

import os
import cv2
import glob
import argparse
from imgaug import augmenters as iaa

def augmentation_images(input_img_dir,img_format,out_img_dir,no_of_replications,rotate_min,rotate_max,scale_min,scale_max,translate_min,translate_max):
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    img_list = glob.glob(os.path.join(input_img_dir,img_format))
    for i in img_list:
        img = cv2.imread(i)
        img_name = i.split('/')[-1]
        mod_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images = []
        for n in range(no_of_replications):
            images.append(mod_img)
        rotate= iaa.Affine(rotate=(rotate_min,rotate_max))
        rot_images_aug = rotate.augment_images(images)
        scale= iaa.Affine(scale=(scale_min,scale_max))
        scaled_images_aug = scale.augment_images(images)
        translate= iaa.Affine(translate_percent=(translate_min,translate_max))
        trans_images_aug = translate.augment_images(images)
        for j in range(len(rot_images_aug)):
            rot_img_name = img_name.split('.')[0] + '_rot_'+str(j)+'.png'
            cv2.imwrite(os.path.join(out_img_dir,rot_img_name),rot_images_aug[j])
        for k in range(len(scaled_images_aug)):
            scale_img_name = img_name.split('.')[0] + '_scale_'+str(k)+'.png'
            cv2.imwrite(os.path.join(out_img_dir,scale_img_name),scaled_images_aug[k])
        for l in range(len(trans_images_aug)):
            translate_img_name = img_name.split('.')[0] + '_translate_'+str(l)+'.png'
            cv2.imwrite(os.path.join(out_img_dir,translate_img_name),trans_images_aug[l])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AUGMENTATION')
    required = parser.add_argument_group('required arguments')
    required.add_argument("--input_img_dir", help="Path of the input image directory", type=str, metavar='', default=None)
    required.add_argument("--img_format", help="Extension of image", type=str, metavar='', default='*.png')
    required.add_argument("--out_img_dir", help="Path of the output augmented image directory", type=str, metavar='', default=None)
    required.add_argument("--no_of_replications", help=" Number of images need to be generated for each type of augmentation", type=str, metavar='', default=2)
    required.add_argument("--rotate_min", help="Minimum value to rotate the image", type=str, metavar='', default=-20)
    required.add_argument("--rotate_max", help="Maximum value to rotate the image.", type=str, metavar='', default=20)
    required.add_argument("--scale_min", help="Minimum value to scale the image", type=str, metavar='', default=0.01)
    required.add_argument("--scale_max", help=" Maximum value to scale the image", type=str, metavar='', default=0.10)
    required.add_argument("--translate_min", help=" Minimum value to scale the image", type=str, metavar='', default=-0.1)
    required.add_argument("--translate_max", help=" Maximum value to scale the image", type=str, metavar='', default=0.1)

    args = parser.parse_args()
    if  args.pdf_path is None or args.input_textfile is None or args.output_dir is None:
        raise TypeError("Supply all 3 arguments to the program!!!")
    augmentation_images(input_img_dir=args.input_img_dir,img_format=args.img_format,out_img_dir=args.out_img_dir,
                        no_of_replications=args.no_of_replications,rotate_min=args.rotate_min,rotate_max=args.rotate_max,
                        scale_min=args.scale_min,scale_max=args.scale_max,translate_min=args.translate_min,translate_max=args.translate_max)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:45:58 2020

@author: developer
"""

import argparse

class Predict:
    
    def __init__():
        pass
    
    def predict_on_single(self):
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this script will be used to predict on test images.')
    parser.add_argument('--sim',type=int, default=1, help="enter the simulation number")
    parser.add_argument('--stage', type=int, default=2, help="enter the stage number")
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='enter the type of trained model')
    args = parser.parse_args()
    

    

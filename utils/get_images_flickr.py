#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:23:02 2020

@author: saugata paul
"""

from datetime import datetime as dt
import flickrapi
import urllib.request as geturl
from PIL import Image
import os
from tqdm import tqdm

global_start = dt.now()

#Creating the initial directory structures
cur_dir = os.getcwd()
train=cur_dir+'/data'+'/train'
val=cur_dir+'/data'+'/validation'
os.mkdir(cur_dir+'/data') if not os.path.isdir(cur_dir+'/data') else None
os.mkdir(train) if not os.path.isdir(train) else None
os.mkdir(val) if not os.path.isdir(val) else None

def get_images(category,n):
    
    #What categories of images you want to download?
    #What's the total of images you want to download?
    #Fraction of images you want for test data? (0 to 1)
    
    #Create a folder corresponding to category name, each category will be stored in a different folder
    os.mkdir(train+"/"+category) if not os.path.isdir(train+"/"+category) else None

    #Using the FlickrAPI key to access flicker 
    flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)
    photos = flickr.walk(text=category,tag_mode='all',tags=category,extras='url_c',per_page=100,sort='relevance')

    #Build a list of valid URLS, we will use these URLs to retrieve images
    url_lists = []
    count=1
    for i, photo in enumerate(photos):
        url = photo.get('url_c')
        if(url!=None):
            url_lists.append(url)
            count+=1

        #Get 'n' valids URLS for 'n' images you want to download
        if count > n:
            break

    #Get training and testing image URLs in two lists        
    train_urls = url_lists

    #This block actually downloads all the 'n'  images belonging to 'keywords' category 
    print("Downloading images...")
    i=1
    for url in tqdm(train_urls):
        folder = train+"/"+category
        geturl.urlretrieve(url, folder+'/{}{}.jpg'.format(category,i))
        image = Image.open(folder+'/{}{}.jpg'.format(category,i)) 
        image = image.resize((256, 256), Image.ANTIALIAS)
        image.save(folder+'/{}{}.jpg'.format(category,i))
        i+=1

    path = folder.split("/")[-3:]
    f_path=""
    for i in path:
        f_path =f_path + "/" + i
        
    print("{} training images of '{}' downloaded and saved in folder '{}'".format(len(train_urls),category,f_path))
    

        
get_images("horse",500)

get_images("cat",500)

get_images("tiger",500)

get_images("monkey",500)

get_images("tree",500)
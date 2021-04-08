# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:27:58 2021

@author: Daniel
"""
print('To run this code you need the following packages:')
print('numpy, pandas, and openCV')
print('if cv2/openCV isnt installed, you can install it with: pip install opencv-python')
import numpy as np
import pandas as pd
import cv2
import os
import time
import pickle
def predict(X,w):
    if X.shape[1]+1 == w.shape[0]:
         X = np.append(X,np.ones((len(X[:,1]),1)),1)
    preds = np.argmax(softmax(np.dot(X,w)),axis=1)
    return preds     
 

def softmax(z):
    sm = (np.exp(z).T/np.sum(np.exp(z),axis=1)).T
    return sm

def oneHot(y):
    YOH = np.zeros((len(y),len(np.unique(y))))
    for idx,value in enumerate(y):
        YOH[idx][value] = 1
    return YOH

#searches for folders within the 
def findpath(pathname=""):
    folders = []
    for name in os.listdir(pathname):
        if(name.endswith(".png")):
            print("This is not a folder.")
        else:
            folders.append(name)
    return folders
    
#Used for checking if the path is valid.
def valid_path(pathname=""):
    try:
        os.listdir(pathname)
        return True
    except: 
        return False

def fileNames(pathname=""):
    images = []
    for filename in os.listdir("{}/".format(pathname)):
        images.append((filename))
    return images

def processing(pathname="",filenames=""):
    feature_vector = []
    for image in filenames:
        #Creates a Black and White version of the imput image.
        im =cv2.imread("{}/{}".format(pathname,image))
        
        #Convert to greyscale
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        #smooth image
        im = cv2.GaussianBlur(im,(5,5),0)
        
        #Close image
        kernel = np.ones((9,9),np.uint8)
        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

        #resize image
        im = cv2.resize(im, (35,35), interpolation = cv2.INTER_LANCZOS4)
        
        #edge detection
        im = cv2.Canny(im,100,200)

        #Convert to array and vectorize/flatten and normalize
        features = np.asarray(im)
        features = features.flatten()
        feature_vector.append(features/255)  
    return feature_vector

if __name__ == "__main__":

    pathname = input("What is the Pathname: ")
    #Checks if this is a valid path. Will continue to ask until its a valid path.
    while True:
        if valid_path(pathname=pathname):
            break
        else:
            pathname = input("That path was invalid, Please enter a new one: ")

    #we have now found if there are folders in the path. Now to extract all the images. 
    filenames = fileNames(pathname=pathname)
    
    print('Loading & Preprocessing Images')
    feature_vector = processing(pathname=pathname,filenames=filenames)
    feature_vector = np.array(feature_vector)
    # target = np.array(target)
    
    print('Predicting with Preprocessed Images')
    file = 'trainedModel'
    loaded_model = pickle.load(open(file, 'rb'))
    start_time = time.time()
    testPred = predict(feature_vector, loaded_model)
    test_time = time.time()-start_time
    print("Test Time: {} seconds".format(test_time))
    #%%
    count1 = np.sum(testPred)
    count0 = len(testPred)-np.sum(testPred)
    out2 = pd.DataFrame([filenames,testPred]).T
    out2 = out2.append(pd.DataFrame(['Count of 0',count0]).T)
    out2 = out2.append(pd.DataFrame(['Count of 1',count1]).T)
    out2.to_excel('Outputs_DN_HH_ZC.xlsx')
    print('Outputs saved to excel file named Outputs_DN_HH_ZC.xlsx')
    # missclassTest1 = testPred!=target
    # missclassRateTest1 = sum(missclassTest1)/target.shape[0]
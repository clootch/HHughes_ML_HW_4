# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:16:57 2021

@author: Daniel
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time
import matplotlib.pyplot as plt
import math
def logisticRegression(X,y,valX,valY,lr,rho):
    #append ones to X for bias weight
    X = np.append(X,np.ones((len(X[:,1]),1)),1)
    valX = np.append(valX,np.ones((len(valX[:,1]),1)),1)
    N = X.shape[0]
    NVal = valX.shape[0]
    #random initialization of w to small weights
    w = np.random.rand(len(X[1,:]),len(np.unique(y)))*0.0001
    #one hot enocde y into Y
    Y_oneHot = oneHot(y)
    Y_oneHotVal = oneHot(valY)
    it = 1000
    losses = []
    lossesVal = []
    v = 0
    numBadEps = 0
    bestLossVal = 10
    ESpatience = 20
    LRpatience = 7
    tol = 0.0001
    LRdelayCount = 0
    LRDelay = 7
    batchSize = 64
    numBatches = math.ceil(N/batchSize)
    for i in range(it):
        #Mini-Batch Grad Descent
        for ii in range(numBatches):
            BX = X[ii*batchSize:(ii+1)*batchSize]
            By = Y_oneHot[ii*batchSize:(ii+1)*batchSize]
            A = softmax(np.dot(BX,w))
            grad = (-1 / N) * np.dot(BX.T,(By - A))
            v = rho*v+lr*grad
            w = w - ( v)
            
        #Full Grad Descent
        # A = softmax(np.dot(X,w))
        # grad = (-1 / N) * np.dot(X.T,(Y_oneHot - A))
        # v = rho*v+lr*grad
        # w = w - ( v)
        #Eval Train and Val sets
        A = softmax(np.dot(X,w))
        loss = (-1 / N) * np.sum(Y_oneHot * np.log(A))
        losses.append(loss)
        
        AVal = softmax(np.dot(valX,w))
        lossVal = (-1 / NVal) * np.sum(Y_oneHotVal * np.log(AVal))
        lossesVal.append(lossVal)
        print(loss)
        print(lossVal)
        
        #Early Stopping & LR scheduler
        

        if  (bestLossVal-tol)<=lossVal:
            numBadEps+=1
            LRdelayCount +=1
        else:
            numBadEps = 0
            bestLossVal = lossVal
        if (numBadEps>LRpatience) & (LRDelay<LRdelayCount):
            lr *=0.1
            LRdelayCount = 0
            print('Reducing LR to',lr)
        if numBadEps>ESpatience:
            break
        print(LRdelayCount)
        print(numBadEps)
        
        
    
    preds = predict(X,w)
    predsVal = predict(valX,w)
    return w,losses,lossesVal,preds,predsVal
    
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

def fileNames(pathname="",folders=""):
    images = []
    for filename in os.listdir("{}/{}".format(pathname,curfolder)):
        images.append((curfolder,filename))
    return images

def processing(pathname="",filenames=""):
    feature_vector = []
    t = []
    for curimageset in filenames:
        for image in curimageset:
            #Creates a Black and White version of the imput image.
            im =cv2.imread("{}/{}/{}".format(pathname,image[0],image[1]))
            
            #Convert to greyscale
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            
            #smooth image
            # kernel = np.ones((6,6),np.float32)/36
            # im = cv2.filter2D(im,-1,kernel)
            im = cv2.GaussianBlur(im,(5,5),0)
            # _,im = cv2.threshold(im,150,255,cv2.THRESH_BINARY)
            # cv2.imshow("{}:{}".format(image[0],image[1]),im)
            # cv2.waitKey(0)
            
            #Close image
            kernel = np.ones((9,9),np.uint8)
            im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow("{}:{}".format(image[0],image[1]),im)
            # cv2.waitKey(0)
            # im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            # cv2.THRESH_BINARY,19,5)
            # cv2.imshow("{}:{}".format(image[0],image[1]),im)
            # cv2.waitKey(0)
            #resize image
            im = cv2.resize(im, (35,35), interpolation = cv2.INTER_LANCZOS4)
            
            
            #edge detection
            
            # cv2.imshow("{}:{}".format(image[0],image[1]),bawi)
            # cv2.waitKey(0)
            im = cv2.Canny(im,100,200)
            # cv2.imshow("{}:{}".format(image[0],image[1]),im)
            # cv2.waitKey(0)

            features = np.asarray(im)
            features = features.flatten()
            feature_vector.append(features/255)
            t.append(int(image[0]))
    return feature_vector,t

if __name__ == "__main__":
    # print(os.listdir("1/"))
    pathname = input("What is the Pathname: ")
    #Checks if this is a valid path. Will continue to ask until its a valid path.
    while True:
        if valid_path(pathname=pathname):
            break
        else:
            pathname = input("That path was invalid, Please enter a new one: ")
    folders = findpath(pathname)
    #we have now found if there are folders in the path. Now to extract all the images. 
    filenames = []
    for curfolder in folders:
        filenames.append(fileNames(pathname=pathname,folders=curfolder))
    #all image names have been extracted, now to do the dirty work. 
    feature_vector,target = processing(pathname=pathname,filenames=filenames)
    feature_vector = np.array(feature_vector)
    target = np.array(target)
        
    np.random.seed(20)
    #Shuffle X and Y
    indicies = [i for i in range(len(target))]
    np.random.shuffle(indicies)
    feature_vector = feature_vector[indicies]
    target = target[indicies]
    
    #Train/Val/Test Split
    
    trainSize = 0.8
    testSize=0.1
    valSize = 1-trainSize-testSize
    N = len(target)
    trainSize = round(trainSize*N)
    testSize = round(testSize*N)
    valSize = round(valSize*N)
    trainX = feature_vector[0:trainSize];trainY = target[0:trainSize]
    testX = feature_vector[trainSize:trainSize+testSize];testY =target[trainSize:trainSize+testSize]
    valX = feature_vector[trainSize+testSize:];valY =target[trainSize+testSize:]
    
    
    
    # for val in feature_vector:
    #     print(len(val))
    
    lr = 0.01
    rho = 0.95
    start_time = time.time()
    w,losses,lossesVal,preds,predsVal = logisticRegression(trainX,trainY,valX,valY,lr,rho)
    train_time = time.time()-start_time
    print("Train Time: {} seconds".format(train_time))
    #Plot losses
    plt.figure()
    plt.plot(losses)
    plt.plot(lossesVal)
    plt.legend(['Train','Val'])
    #Calc missClassRates
    missclassTrain = preds!=trainY
    missclassRateTrain = sum(missclassTrain)/trainY.shape[0]
    missclassVal = predsVal!=valY
    missclassRateVal = sum(missclassVal)/valY.shape[0]
    #%%
    #Test on Test Set
    start_time = time.time()
    testPred = predict(testX, w)
    test_time = time.time()-start_time
    print("Test Time: {} seconds".format(test_time))
    missclassTest = testPred!=testY
    missclassRateTest = sum(missclassTest)/testY.shape[0]
    
    from sklearn.metrics import confusion_matrix
    cfm = confusion_matrix(testY,testPred)
    print(cfm)
#%%
    import pickle
    file = 'trainedModel'
    pickle.dump(w, open(file, 'wb'))
    loaded_model = pickle.load(open(file, 'rb'))
    start_time = time.time()
    testPred = predict(testX, loaded_model)
    test_time = time.time()-start_time
    print("Test Time: {} seconds".format(test_time))
    missclassTest1 = testPred!=testY
    missclassRateTest1 = sum(missclassTest1)/testY.shape[0]

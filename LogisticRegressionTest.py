# -*- coding: utf-8 -*-
#%% Test Logistic Regression implementation
import numpy as np
import matplotlib.pyplot as plt
#%% Logistic Regression


def logisticRegression(X,y,lr):
    #append ones to X for bias weight
    X = np.append(X,np.ones((len(X[:,1]),1)),1)
    N = X.shape[0]
    #random initialization of w to small weights
    w = np.random.rand(len(X[1,:]),len(np.unique(y)))*0.01
    #one hot enocde y into Y
    Y_oneHot = oneHot(y)
    
    it = 1000
    losses = []
    for i in range(it):
        #find X.w
        A = softmax(np.dot(X,w))
        loss = (-1 / N) * np.sum(Y_oneHot * np.log(A))
        losses.append(loss)
        grad = (-1 / N) * np.dot(X.T,(Y_oneHot - A))
        w = w - (lr * grad)
    
    preds = np.argmax(softmax(np.dot(X,w)),axis=1)
    return w,losses,preds
    


def softmax(z):
    sm = (np.exp(z).T/np.sum(np.exp(z),axis=1)).T
    return sm

def oneHot(y):
    YOH = np.zeros((len(y),len(np.unique(y))))
    for idx,value in enumerate(y):
        YOH[idx][value] = 1
    return YOH

if __name__ == "__main__":
    #define 2-D means and stds of 3 different classes
    means1 = [1,0]
    stds1 = [0.5,0.5]
    means2 = [2,2]
    stds2 = [0.5,0.5]
    means3 = [0,2]
    stds3 = [0.5,0.5]
    # Number of samples per class
    numSamples = 50
    
    np.random.seed(1)
    #Create data for 3 classs
    Class1X1 = np.random.normal(means1[0],stds1[0],numSamples)
    Class1X2 = np.random.normal(means1[1],stds1[1],numSamples)
    class1 = np.stack([Class1X1,Class1X2],axis=1)
    Class2X1 = np.random.normal(means2[0],stds2[0],numSamples)
    Class2X2 = np.random.normal(means2[1],stds2[1],numSamples)
    class2 = np.stack([Class2X1,Class2X2],axis=1)
    Class3X1 = np.random.normal(means3[0],stds3[0],numSamples)
    Class3X2 = np.random.normal(means3[1],stds3[1],numSamples)
    class3 = np.stack([Class3X1,Class3X2],axis=1)
    #Concat all classes data to form X and Y
    X = np.concatenate((class1,class2,class3))
    Y = np.concatenate((np.zeros(len(class1),dtype=int),np.ones(len(class2),dtype=int),np.ones(len(class3),dtype=int)*2))
    
    #Plot X on graph
    fig, ax = plt.subplots()
    ax.scatter(class1[:,0], class1[:,1], c='Red', label='Class1', edgecolors='none')
    ax.scatter(class2[:,0], class2[:,1], c='Blue', label='Class2', edgecolors='none')
    ax.scatter(class3[:,0], class3[:,1], c='green', label='Class3', edgecolors='none')
    ax.legend()
    
    #Shuffle X and Y
    indicies = [i for i in range(len(Y))]
    np.random.shuffle(indicies)
    X = X[indicies]
    Y = Y[indicies]

    # Use multi-class Logistic Regression for classification on training data
    w,losses,preds = logisticRegression(X,Y,0.5)
    # Plot losses vs epoch
    plt.figure()
    plt.plot(losses)
    
    #Find missclassified samples
    missclass = preds!=Y
    missclassRate = sum(missclass)/Y.shape[0]
    ax.scatter(X[missclass,0],X[missclass,1],c='black',marker='x',linewidths=1)
    
    
    
    
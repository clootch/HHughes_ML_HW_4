import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time
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
    for curimageset in filenames:
        for image in curimageset:
            #Creates a Black and White version of the imput image.
            im =cv2.imread("{}/{}/{}".format(pathname,image[0],image[1]))
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            _,bawi = cv2.threshold(im,150,255,cv2.THRESH_BINARY)
            cv2.imshow("{}:{}".format(image[0],image[1]),bawi)
            cv2.waitKey(0)
            features = np.asarray(bawi)
            features = features.flatten()
            feature_vector.append(features)
    return feature_vector  

if __name__ == "__main__":
    print(os.listdir("E:\Visual Studio Code (Python and CPP)\Python Code\Machine Learning\Homework 4\images"))
    pathname = input("What is the Pathname: ")
    #Checks if this is a valid path. Will continue to ask until its a valid path.
    while True:
        if valid_path(pathname=pathname):
            break
        else:
            pathname = input("That path was invalid, Please enter a new one: ")
    pathname = "E:\Visual Studio Code (Python and CPP)\Python Code\Machine Learning\Homework 4\images"
    folders = findpath(pathname)
    #we have now found if there are folders in the path. Now to extract all the images. 
    filenames = []
    for curfolder in folders:
        filenames.append(fileNames(pathname=pathname,folders=curfolder))
    #all image names have been extracted, now to do the dirty work. 
    feature_vector = processing(pathname=pathname,filenames=filenames)
    print(len(feature_vector))
    for val in feature_vector:
        print(len(val))
    start_time = time.time()
    print("Execution Time: {} seconds".format(time.time()-start_time))
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

def filenames(pathname="",folders=""):
    images = []
    temp = []
    for curfolder in folders: 
        for filename in os.listdir("{}/{}".format(pathname,curfolder)):
            temp.append(filename)
        images.append(temp)
        temp = []
    return images

def processing(pathname="",curfolder="",filenames=""):
    for curimageset in filenames:
        for image in curimageset:
            #Creates a Black and White version of the imput image.
            im =cv2.imread("{}/{}/{}".format(pathname,curfolder,image))
            _,bawi = cv2.threshold(im,150,255,cv2.THRESH_BINARY)
            cv2.imshow("Black and White",bawi)
            features = np.asarray(bawi)
            features = features.flatten()
            print(len(features))
            cv2.waitKey(0)

if __name__ == "__main__":
    
    pathname = input("What is the Pathname: ")
    #Checks if this is a valid path. Will continue to ask until its a valid path.
    while True:
        if valid_path(pathname=pathname):
            break
        else:
            pathname = input("That path was invalid, Please enter a new one: ")
    pathname = "E:\Visual Studio Code (Python and CPP)\Python Code\Machine Learning\Homework 4"
    folders = findpath(pathname)
    #we have now found if there are folders in the path. Now to extract all the images. 
    filenames = filenames(pathname=pathname,folders=folders)
    #all image names have been extracted, now to do the dirty work. 
    for curfolder in folders:
        processing(pathname=pathname,curfolder=curfolder,filenames=filenames)
    start_time = time.time()
    print("Execution Time: {} seconds".format(time.time()-start_time))
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time
#Need to find what quantifiers I want for classification. 
"""
So if i use CV2 to filter out all of the crap, then find the area of the white space in the image (Anything of importance)
then I could use that to classify if its a worm or not.
"""

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


if __name__ == "__main__":
    start_time = time.time()
    pathname = input("What is the Pathname: ")
    #Checks if this is a valid path. Will continue to ask until its a valid path.
    while True:
        if valid_path(pathname=pathname):
            break
        else:
            pathname = input("That path was invalid, Please enter a new one: ")
    pathname = "D:/Celegans_Train"
    folders = findpath(pathname)
    #we have now found if there are folders in the path. Now to extract all the images. 
    filenames = filenames(pathname=pathname,folders=folders)
    #all image names have been extracted, now to do the dirty work. 
    print("Execution Time: {}".format(time.time()-start_time))
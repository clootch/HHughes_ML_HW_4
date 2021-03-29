import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import sys
def test(pathname="",model=""):
    i = 0
    #Tests if given path works, closes if not. (Make this recursive to keep grabbing
    #inputs until works)
    try:
        directories=os.listdir(pathname)
    except:
        print("Path given was invalid. Ending program...")
        sys.exit(0)
    #Catches any invalid paths to walk into
    for val in directories:
        try:
            os.listdir("{}/{}".format(pathname,val))
        except:
            print("Invalid Path. Removing {} From Directory...".format(val))
            directories.pop(i)
        i += 1
    images = []
    #Grabs all the image path names from the directories
    for directory in directories:
        for imageName in os.listdir("{}/{}".format(pathname,directory)):
            images.append(imageName)
    #Now we just have the images pathnames, time to do the real work. 
    

    
pathname = input("Enter the Pathname: ")
pathname = "E:/Visual Studio Code (Python and CPP)/Python Code/Machine Learning/Homework 4"
test(pathname=pathname)
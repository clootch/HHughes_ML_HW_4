import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

#Need to find what quantifiers I want for classification. 
"""
So if i use CV2 to filter out all of the crap, then find the area of the white space in the image (Anything of importance)
then I could use that to classify if its a worm or not.
"""

def findpath(pathname=""):
    for name in os.listdir(pathname):
        print(name)
    
def valid_path(pathname=""):
    try:
        os.listdir(pathname)
        return True
    except: 
        return False

if __name__ == "__main__":
    pathname = input("What is the Pathname: ")
    done = False
    while True:
        if valid_path(pathname=pathname):
            break
        else:
            pathname = input("That path was invalid, Please enter a new one: ")
    pathname = "D:/Celegans_Train"
    findpath(pathname)
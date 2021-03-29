import cv2
import os
import numpy as np
import time
for val in range(15):
    start_time = time.time()
    im = cv2.imread("E:/Visual Studio Code (Python and CPP)/Python Code/Machine Learning/Homework 4/1/image_{}.png".format(val+540))
    #im = cv2.resize(im,(200,200))
    #edge = cv2.Canny(im,150,200,255)
    _,bawi = cv2.threshold(im,200,255,cv2.THRESH_BINARY)
    edge = cv2.Canny(bawi,150,200,255)
    #cv2.imshow("Image_{}".format(val+494),im)
    #cv2.imshow("BAWI_Image_{}".format(val+494),bawi)
    cv2.imshow("Edged_Image_{}".format(val+494),edge)
    cv2.waitKey(0)
cv2.destroyAllWindows()
print("{} Seconds for execution".format(time.time()-start_time))

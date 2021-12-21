import cv2
import numpy as np
import pandas as pd
import os

train_dict = {
    "clean": ["rubbish_detection/train/clean/" + str(filename) for filename in os.listdir("rubbish_detection/train/clean")],
    "dirty": ["rubbish_detection/train/dirty/" + str(filename) for filename in os.listdir("rubbish_detection/train/dirty")]
}

test_dict = {
    "clean": ["rubbish_detection/test/clean/" + str(filename) for filename in os.listdir("rubbish_detection/test/clean")],
    "dirty": ["rubbish_detection/test/dirty/" + str(filename) for filename in os.listdir("rubbish_detection/test/dirty")]
}

label_dict = {
    "clean": 0,
    "dirty": 1
}

def get_edges(img):
    new_img = img.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.GaussianBlur(new_img,(5,5),0)
    new_img = cv2.Canny(new_img, 167, 58)
    
    # low = np.array([167])
    # high = np.array([58])
    
    # mask = cv2.inRange(new_img,high,low)
    new_img = cv2.bitwise_or(img,img,mask=new_img)
    
    return new_img

img = cv2.imread(str(train_dict['dirty'][312]))
cv2.imshow("Original", img)
cv2.imshow("Image", get_edges(img))
cv2.waitKey(0)

# 167, 58



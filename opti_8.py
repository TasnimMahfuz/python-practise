# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:03:54 2024

@author: Nafis
"""

import numpy as np
import cv2 as cv
import argparse



cap = cv.VideoCapture(0)

# Check if webcam initialization was successful
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
print(type(p0))

print(p0)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:28:33 2024

@author: nafis
"""

import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('--image', type=str, help='path to image file')
args = parser.parse_args()

# Use webcam by default if no image path is provided
if args.image:
    cap = cv.VideoCapture(args.image)
else:
    cap = cv.VideoCapture(0)

# Check if webcam initialization was successful
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Function to

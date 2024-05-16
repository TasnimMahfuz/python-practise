# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:58:40 2024

@author: Nafis
"""

import numpy as np
import cv2 as cv
import argparse
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#if args.image:
 #   cap = cv.VideoCapture(args.image)
#else:
cap = cv.VideoCapture(0)

# Check if webcam initialization was successful
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


ret, old_frame = cap.read()
old_gray = old_frame
#old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

def get_hand_coordinates(old_gray):
    hand_coordinates = []
    #cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        #while cap.isOpened():
            #success, image = cap.read()
            #if not success:
                #print("Ignoring empty camera frame.")
                #continue

            #image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            #image.flags.writeable = False
            #results = hands.process(image)
            #image.flags.writeable = True
            results = hands.process(old_gray)
            #image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            old_gray = cv.cvtColor(old_gray, cv.COLOR_RGB2BGR)

            while not results.multi_hand_landmarks:
                print("waiting.....")
                
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_coords = []
                    for landmark in hand_landmarks.landmark:
                        cx, cy = landmark.x * old_gray.shape[1], landmark.y * old_gray.shape[0]
                        hand_coords.append([cx, cy])
                    hand_coordinates.append(hand_coords)
                    return np.array(hand_coordinates)


parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('--image', type=str, help='path to image file')
args = parser.parse_args()

# Use webcam by default if no image path is provided
#if args.image:
 #   cap = cv.VideoCapture(args.image)
#else:
 #   cap = cv.VideoCapture(0)


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

#p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
p0 = get_hand_coordinates(old_gray)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#p1 = cv.goodFeaturesToTrack()
print(p0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    print(p0)
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()

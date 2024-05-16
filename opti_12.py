import numpy as np
import cv2 as cv
import argparse
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, old_frame = cap.read()
old_gray = old_frame

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('--image', type=str, help='path to image file')
args = parser.parse_args()

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

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect hand landmarks
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        results = hands.process(old_gray)
        while not results.multi_hand_landmarks:
            print("waiting.....")
            results = hands.process(old_gray)

        hand_coordinates = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    cx, cy = landmark.x * old_gray.shape[1], landmark.y * old_gray.shape[0]
                    hand_coordinates.append([cx, cy])

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, np.array(hand_coordinates), None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = np.array(hand_coordinates)[st == 1]

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

    # Update the previous frame
    old_gray = frame_gray.copy()

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()

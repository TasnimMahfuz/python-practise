import numpy as np
import cv2 as cv
import argparse

from PIL import Image

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
cv.imwrite("old_frame.jpg",old_frame)
#old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

img = Image.OPEN(old_frame.jpg)
width, height = img.size()

left = width/2
right = width/2
top = height/2
bottom = height/2

img2 = img.crop(left,top,right, bottom)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(img2, mask=None, **feature_params)
p1 = cv.goodFeaturesToTrack()
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


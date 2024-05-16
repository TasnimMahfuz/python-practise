import cv2 as cv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hand_coor = []
cap = cv.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("camera frame not found")
            continue

        image = cv.cvtColor(cv.flip(image,1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image_height, image_width, _ = image.shape

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy = landmrk.x*image_width, landmrk.y*image_height
                    hand_coor.append([cx,cy])

            print(hand_coor)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv.imshow('MediaPipe Hands', image)
            cv.imwrite('an_image.jpg', image)
            break;

feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

lk_params = dict( winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255, (100,3))

ret,old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

#p0 = hand_coor;p0 = np.array(hand_coor, dtype=np.float32)
p0 = np.array(hand_coor,dtype=np.float32)

mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("There was a video feed reading failure...")
        break;

    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        #good_new = p1[st == 1]
        #good_old = p0[st == 1]
        good_new = p1[st[:,0] == 1]
        good_old = p0[st[:,0] == 1]

    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a, b= new.ravel()
        c, d = old.ravel()

        #mask = cv.line(mask, (int(a), int(b), int(c), int(d)), color[i].tolist(),2)
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #frame = cv.circle(frame, (int(a),int(b), 5, color[i].tolist(), -1))
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv.add(frame, mask)

    cv.imshow('frame',img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break;
    old_gray = frame_gray.copy()
    #p0 = good_new.reshape(1,1,2)
    p0 = good_new.reshape(-1, 1, 2)


cap.release()
cv.destroyAllWindows()

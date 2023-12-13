import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


#Get MP holistics
mp_holistic = mp.solutions.holistic #holistic model for making detections
mp_drawing = mp.solutions.drawing_utils #drawing utilities for drawing them

def mediapipe_detection(image, model): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     #by default, opencv reads video feed in BGR. for mediapipe detection, need RGB
    image.flags.writeable = False
    results = model.process(image) #this is where detection takes place. image is frame from opencv
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #unprocess image
    return image, results


#Get keypoints using MP (Access to webcam)
vid = cv2.VideoCapture(0) #could substitute 0 with name of video input

#Set mediapipe model *
with mp_holistic.Holistic(min_detection_confidence =0.5, min_tracking_confidence=0.5) as holistic:
    while vid.isOpened():
        #read each frame
        ret, frame = vid.read()

        #make detections
        image,results = mediapipe_detection(frame, holistic)
        

        #show each frame
        cv2.imshow('OpenCV Feed', frame)

        #breaking out of our video loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#release our webcam
vid.release()
cv2.destroyAllWindows()

print(results.face_landmarks) #face detection. will return error if no face in frame



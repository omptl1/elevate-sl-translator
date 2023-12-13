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

#Visualize detection landmarks using results from mediapipe model
def draw_landmarks(image, results): 
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) #CONNECTIONS is basically a connection map connecting landmarks within the frame
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #mp_drawing.draw_landmarks modifies the current frame with the landmarks applied to it
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #color in rgb
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


#Get keypoints using MP (Access to webcam)
vid = cv2.VideoCapture(0) #could substitute 0 with name of video input

#Set mediapipe model *
with mp_holistic.Holistic(min_detection_confidence =0.5, min_tracking_confidence=0.5) as holistic:
    while vid.isOpened():
        #read each frame
        ret, frame = vid.read()

        #make detections
        image,results = mediapipe_detection(frame, holistic)
        print(results)


        #draw our landmarks
        draw_styled_landmarks(image, results)
        

        #render each image (image is frame processed and landmarks applied)
        cv2.imshow('OpenCV Feed', image)

        #breaking out of our video loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#release our webcam
vid.release()
cv2.destroyAllWindows()


from calendar import LocaleHTMLCalendar
from collections import Counter
from decimal import localcontext
import cv2
import sys
#import RPi.GPIO as GPIO
import mediapipe as mp
import numpy as np
import time
import os
import pyautogui

#setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

#window_pos_x = 240 
#window_pos_y = 120
window_width = 640
window_height = 480
capture_frame = 4

# Select exercise menu
menu_num = 0 
menu_time = 10 #selecting time(window =30, rasp= )
menu_num_count = 0
menu_num_temp1 = 0
menu_num_temp2 = 0
global ex_count
ex_count = [0,0,0,0] #if you change count para, final result screen also change
# Curl counter variables
counter = [0,0,0,0] 
lcounter=[0,0,0,0]
rcounter =[0,0,0,0]
stage1 = None
stage2 = None
stage = None

BUTTON_GPIO = 16
mpbreak = 0
def initialize_var():
    global menu_num, menu_num_count, menu_num_temp1, menu_num_temp2,counter,lcounter, rcounter, stage1, stage2, stage, per, bar
    menu_num = 0 
    menu_num_count = 0
    menu_num_temp1 = 0
    menu_num_temp2 = 0
    counter[0:3] = 0 
    lcounter[0:3]=0
    rcounter[0:3] =0
    stage1 = None
    stage2 = None
    stage = None
    per = 0
    bar = 0
    
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
    
menu_status = [" -","Dumbell","Lunge","Squat","PushUp","Exit","--","--","--","--","--","--"]

def findPosition(image, draw=True):
  lmList = []
  if results.pose_landmarks:
      mp_drawing.draw_landmarks(
         image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      for id, lm in enumerate(results.pose_landmarks.landmark):
          h, w, c = image.shape
          cx, cy = int(lm.x * w), int(lm.y * h)
          lmList.append([id, cx, cy])
          #cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
  return lmList

'''
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
global pressed
pressed = False      
'''

# For webcam input:

cv2.namedWindow('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
cap.set(cv2.CAP_PROP_FPS, capture_frame)
while(1):
    # Select menu with hand
    with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        image = cv2. flip(image, 1)
        if not success:
          print("Ignoring empty camera frame.")
          continue
        '''
        # button is pressed when pin is LOW
        if not GPIO.input(BUTTON_GPIO):
            if not pressed:
                print("Button pressed in menu!")
                initialize_var()
                menu_num = 5
                pressed = True
                break
        # button not pressed (or released)
        else:
            pressed = False
        '''
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initially set finger count to 0 for menu select
        fingerCount = 0

        if results.multi_hand_landmarks:

          for hand_landmarks in results.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            # Set variable to keep landmarks positions (x and y)
            handLandmarks = []

            # Fill list with x and y positions of each landmark
            for landmarks in hand_landmarks.landmark:
              handLandmarks.append([landmarks.x, landmarks.y])

            # Test conditions for each finger: Count is increased if finger isconsidered raised.
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
              fingerCount = fingerCount+1
            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
              fingerCount = fingerCount+1

            # Other fingers: TIP y position must be lower than PIP y position as image origin is in the upper left corner.
            if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
              fingerCount = fingerCount+1
            if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
              fingerCount = fingerCount+1
            if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
              fingerCount = fingerCount+1
            if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
              fingerCount = fingerCount+1

            # Draw hand landmarks 
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Display menu
        cv2.rectangle(image, (0,0), (400,73), (185,245,16), -1)
        
        cv2.putText(image, 'Select', (20,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(fingerCount), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        cv2.putText(image, 'Menu', (200,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(menu_status[fingerCount]),(150,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        menu_num_temp1 = fingerCount
        menu_num_count += 1
        if(menu_num_count > menu_time) :
            menu_num_temp2 = fingerCount
            menu_num_count = 0
        if(menu_num_temp1 == menu_num_temp2):
            menu_num = fingerCount
            menu_num_temp1 = 0
            menu_num_temp2 = 0
            menu_num_count = 0
                    
        print(menu_num)
        
        # Display image
        cv2.namedWindow('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #image = cv2.resize(image,dsize=(0,0), fx = 1.5,fy=1.5, interpolation = cv2.INTER_LINEAR)
        #image = cv2.resize(image,dsize=(1440,900),interpolation = cv2.INTER_CUBIC)
        cv2.imshow('MediaPipe Menu', image)
        
        if (cv2.waitKey(5) & 0xFF == 27) or (menu_num != 0):
          menu_num_count = 0
          break
      
      
        
    if (menu_num == 1) :
      print("Menu 1")
      with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            #
            frame = cv2.flip(frame,1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                lshoulder= [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                rshoulder= [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                langle = int(calculate_angle(lshoulder, lelbow, lwrist))
                rangle = int(calculate_angle(rshoulder, relbow, rwrist))
                # Visualize angle
                cv2.putText(image, str(langle), 
                               tuple(np.multiply(lelbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 5, cv2.LINE_AA
                                    )

                cv2.putText(image, str(rangle), 
                               tuple(np.multiply(relbow, [540, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 5, cv2.LINE_AA
                                    )
                cv2.putText(image, str(langle), 
                               tuple(np.multiply(lelbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,20,20), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(rangle), 
                               tuple(np.multiply(relbow, [540, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,20,20), 2, cv2.LINE_AA
                                    )
                global bar
                if langle < rangle:
                    per = np.interp(langle, (20,165),(100,0))
                    bar = np.interp(langle, (15,165),(120,0))
                else :
                    per = np.interp(rangle, (20,165),(100,0))
                    bar = np.interp(rangle, (15,165),(120,0))
                
                # Curl counter logic
                if langle > 152:
                    stage1 = "left down"
                if rangle > 152:
                    stage2 = "right down"
                if (langle < 35 and stage1 =='left down'):
                    stage1="left up"
                    stage = "left up"
                    lcounter[0] +=1
                    counter[0] = lcounter[0] + rcounter[0]
                    print(counter[0])
                if (rangle < 35 and stage2 =='right down'):
                    stage2="right up"
                    stage = "right up"
                    rcounter[0] +=1
                    counter[0] = lcounter[0] + rcounter[0]
                    print(counter[0])
                if (langle > 130) and (rangle > 130):
                    stage = "down"

            except:
                pass
            
            # Percentage bar
            cv2.rectangle(image, (40,300), (70,420), (255,255,255), cv2.FILLED)
            cv2.rectangle(image, (40,420-int(bar)), (70,420), (130,45,216), cv2.FILLED)
            cv2.rectangle(image, (40,300), (70, 420), (120,120,120), 2)
            cv2.putText(image, f'{int(per)}%', (30,280),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255,255), 16, cv2.LINE_AA)
            cv2.putText(image, f'{int(per)}%', (30,280),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 80,180), 3, cv2.LINE_AA)
            
            # Setup status box
            cv2.rectangle(image, (0,0), (360,80), (85,45,116), -1)
           
            # Rep data
            cv2.putText(image, 'COUNT', (15,23),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(counter[0]), (15,63), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            # Stage data
            cv2.putText(image, 'STAGE', (160,23), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (160,63), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               
                
            cv2.namedWindow('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('MediaPipe Menu', image)
            #cv2.moveWindow('MediaPipe Menu',  window_pos_x, window_pos_y)
        
            if cv2.waitKey(10) & 0xFF == ord('r'): #pose로 수정필요 
                counter[0] = 0
                lcounter[0] = 0
                rcounter[0] = 0
                stage = "Reset"
            if cv2.waitKey(10) & 0xFF == 27:
                menu_num=0
                ex_count[0] = counter[0] #also need to change GPIO
                break
                '''
            if not GPIO.input(BUTTON_GPIO):
                if not pressed:
                    print("Button pressed in 1 !")
                    result_1 = counter 
                    initialize_var()
                    pressed = True
                    break
            # button not pressed (or released)
            else:
                pressed = False
        '''
            
    if (menu_num == 2) :
        print("Menu 2")
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.flip(frame,1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    lhip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    rhip= [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


                    # Calculate angle
                    langle = int(calculate_angle(lhip, lknee, lankle))
                    rangle = int(calculate_angle(rhip, rknee, rankle))

                    # Visualize angle
                    cv2.putText(image, str(langle), 
                                   tuple(np.multiply(lknee, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(image, str(rangle), 
                                   tuple(np.multiply(rknee, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    
                    if langle < rangle:
                        per = np.interp(langle, (110,160),(100,0))
                        bar2 = np.interp(langle, (100,160),(120,0))
                    else :
                        per = np.interp(rangle, (110,160),(100,0))
                        bar2 = np.interp(rangle, (100,160),(120,0))

                    # lunge counter logic
                    if (langle > 160 and rangle > 160):
                        stage1 = "left up"
                        stage2 = "right up"
                    if (langle < 110 and stage1 =='left up'):
                        stage1 = "left down"
                        stage  = "left down"
                        lcounter[1] +=1
                        counter[1] = lcounter[1] + rcounter[1]
                        print(counter[1])
                    if (rangle < 110 and stage2 =='right up'):
                        stage2 = "right down"
                        stage  = "right down"
                        rcounter[1] +=1
                        counter[1] = lcounter[1] + rcounter[1]
                        print(counter[1])
                    if (langle > 160) and (rangle > 160):
                        stage = "up"

                except:
                    pass

                # Percentage bar
                cv2.rectangle(image, (40,300), (70,420), (255,255,255), cv2.FILLED)
                cv2.rectangle(image, (40,420-int(bar2)), (70,420), (255,0,1), cv2.FILLED)
                cv2.rectangle(image, (40,300), (70, 420), (120,120,120), 2)
                cv2.putText(image, f'{int(per)}%', (30,280),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 16, cv2.LINE_AA)
                cv2.putText(image, f'{int(per)}%', (30,280),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,216,0), 3, cv2.LINE_AA)

                # Setup status box
                cv2.rectangle(image, (0,0), (360,80), (255,216,0), -1)

                # Rep data
                cv2.putText(image, 'Count', (15,23), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter[1]), 
                            (15,63), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

                # Stage data
                cv2.putText(image, 'STAGE', (160,23), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (160,63), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)


                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )               

                cv2.imshow('MediaPipe Menu', image)

                if cv2.waitKey(10) & 0xFF == ord('r'): #pose로 수정필요 
                    counter[1] = 0
                    lcounter[1] = 0
                    rcounter[1] = 0
                    stage = "Reset"
                if cv2.waitKey(10) & 0xFF == 27:
                    menu_num=0
                    ex_count[1] = counter[1]
                    break
                    '''
                if not GPIO.input(BUTTON_GPIO):
                    if not pressed:
                        print("Button pressed in 2 !")
                        initialize_var()
                        pressed = True
                        break
                # button not pressed (or released)
                else:
                    pressed = False                
                    '''

    if (menu_num == 3) :
        print("Menu 3")
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
          while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                lhip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                rhip= [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angle
                langle = int(calculate_angle(lhip, lknee, lankle))
                rangle = int(calculate_angle(rhip, rknee, rankle))
                # Visualize angle
                cv2.putText(image, str(langle), 
                               tuple(np.multiply(lknee, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(rangle), 
                               tuple(np.multiply(rknee, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                # Sqaut counter logic
                if (langle < 120 and rangle < 120):
                    stage = "down"

                if (langle > 160 and rangle > 160 and stage =='down'):
                    stage = "up"
                    counter[2] +=1
                    print(counter[2])


            except:
                pass

            # Setup status box
            cv2.rectangle(image, (0,0), (350,73), (185,245,16), -1)

            # Rep data
            cv2.putText(image, 'Count', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter[2]), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (160,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               

            cv2.namedWindow('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('MediaPipe Menu', image)

            if cv2.waitKey(10) & 0xFF == ord('r'): #pose로 수정필요 
                counter = 0
                lcounter = 0
                rcounter = 0
                stage = "Reset"
                
            if cv2.waitKey(10) & 0xFF == 27:
                menu_num=0
                ex_count[2] = counter

                break
        '''
            if not GPIO.input(BUTTON_GPIO):
                if not pressed:
                    print("Button pressed in 3 !")
                    initialize_var()
                    pressed = True
                    break
            # button not pressed (or released)
            else:
                pressed = False       
                    '''
    if (menu_num == 4) :
        print("Menu 4")
        with mp_pose.Pose(min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as pose:

          while cap.isOpened():

            success, image = cap.read()
            image = cv2.flip(image, 1)
            #image = cv2.resize(image, (640,480))

            if not success:

              print("Ignoring empty camera frame.")

              # If loading a video, use 'break' instead of 'continue'.

              continue

            # Flip the image horizontally for a later selfie-view display, and convert

            # the BGR image to RGB.

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to

            # pass by reference.

            results = pose.process(image)

            # Draw the pose annotation on the image.

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            lmList = findPosition(image, draw=True)

            if len(lmList) != 0:

              cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)

              cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)

              cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)

              cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)

              if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):

                cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)

                cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)

                stage = "down"

              if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":

                stage = "up"

                counter += 1

                counter2 = str(int(counter))

                print(counter)

                os.system("echo '" + counter2 + "' | festival --tts")

            text = "{}:{}".format("Push Ups", counter)

            cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,

                        1, (255, 0, 0), 2)

            cv2.namedWindow('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('MediaPipe Menu', image)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop

            if key == ord("q"):
              menu_num=0
              break
              
            if cv2.waitKey(10) & 0xFF == 27:
                menu_num=0
                ex_count[3] = counter
                break
                '''
            if not GPIO.input(BUTTON_GPIO):
                if not pressed:
                    print("Button pressed in 4 !")
                    pressed = True
                    initialize_var()
                    break
            # button not pressed (or released)
            else:
                pressed = False       
                '''
    if (menu_num == 5) :
        result_x = 120
        result_y = [0,0,0,0]
        result_list = [140, 220, 300, 380]
        result_count = 0
        for j in range(4):
            if(ex_count[j] != 0):
                result_y[j] = result_list[result_count]
                result_count +=1

        while(cap.isOpened()):
            success, image = cap.read()
            image = cv2. flip(image, 1)
            # result box
            cv2.rectangle(image, (80,80), (560,400), (30,30,30), -1) #width 480 height 320 start pt 80 80 
            # Result data
            cv2.putText(image, 'RESULT', (160,100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(image, 'COUNT', (380,100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 3, cv2.LINE_AA)
            for i in range(4):
                if(ex_count[i] != 0):
                    cv2.putText(image, str(menu_status[i+1]), (result_x,result_y[i]), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(ex_count[i]), (result_x+270,result_y[i]), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Menu', image)
            if cv2.waitKey(1) & 0xFF == 27:
                menu_num=0
                break
            mpbreak  += 1
            if mpbreak > 60: #window 60 but pi= 20?
                mpbreak = 0
                break
            
        print("Magic Mirror On")
        pyautogui.hotkey('alt', 'tab', interval=0.01)
        cap.release()
        cv2.destroyAllWindows()
        initialize_var()
        '''
        while(1):
        
                if not GPIO.input(BUTTON_GPIO):
                    if not pressed:
                        print("Button pressed in 5 !")
                        pressed = True
                        break
                # button not pressed (or released)
                else:
                    pressed = False         
                time.sleep(1)
                             
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
        cap.set(cv2.CAP_PROP_FPS, capture_frame)
        cv2.namedWindow('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        '''
        break

import cv2
import mediapipe as mp
import numpy as np  #배열지원
import time
import os

#setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose #pose 측정

window_pos_x = 240
window_pos_y = 120

menu_num = 0 # Select exercise menu
menu_time = 30
menu_num_count = 0
menu_num_temp1 = 0
menu_num_temp2 = 0

counter = 0 # Curl counter variables
lcounter=0
rcounter =0
stage1 = None
stage2 = None
stage = None

# angle func
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle
# menu func
def menu_status(status):
    match status:
        case 0: #nothing
            return " -"
        case 1: #덤벨
            return "Dumbell"
        case 2: #lunge
            return "lunge"
        case 3: #Plank
            return "Plank"
        case 4: #PushUp
            return "PushUp"
        case 5: #exit
            return "Exit"
        case _: #default
            return "--"


# For webcam input:
cap = cv2.VideoCapture(0)

while(menu_num != 5):


    # Select menu with hand
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.") # If loading a video, use 'break' instead of 'continue'.
          continue

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
        cv2.putText(image, str(menu_status(fingerCount)),(150,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        menu_num_temp1 = fingerCount
        menu_num_count += 1
        if(menu_num_count >= menu_time) :
            menu_num_temp2 = fingerCount
            menu_num_count = 0

        if(menu_num_temp1 == menu_num_temp2):
            menu_num = fingerCount
            menu_num_temp1 = 0
            menu_num_temp2 = 0
            menu_num_count = 0


        print(menu_num)

        # Display image
        cv2.namedWindow('MediaPipe Menu')
        cv2.imshow('MediaPipe Menu', image)
        cv2.moveWindow('MediaPipe Menu', window_pos_x, window_pos_y)

        if (cv2.waitKey(5) & 0xFF == 27) or (menu_num != 0):
          #cv2.destroyAllWindows()
          menu_num_count = 0
          break

    if (menu_num == 1) :
        print("Menu 1")
        cv2.imshow('MediaPipe Menu', image)

    if (menu_num == 2) :
      print("Menu 2")
      #cap = cv2.VideoCapture(0)

      with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

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

                # lunge counter logic
                if langle > 140:
                    stage1 = "left up"
                if rangle > 140:
                    stage2 = "right up"
                if (langle < 110 and stage1 =='left up'):
                    stage1="left down"
                    stage = "left down"
                    lcounter +=1
                    counter = lcounter + rcounter
                    print(counter)
                if (rangle < 110 and stage2 =='right up'):
                    stage2="right down"
                    stage = "right down"
                    rcounter +=1
                    counter = lcounter + rcounter
                    print(counter)
                if (langle > 140) and (rangle > 140):
                    stage = "up"

            except:
                pass

            # Setup status box
            cv2.rectangle(image, (0,0), (350,73), (185,245,16), -1)

            # Rep data
            cv2.putText(image, 'Count', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
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

            cv2.imshow('MediaPipe Menu', image)
            cv2.moveWindow('MediaPipe Menu',  window_pos_x, window_pos_y)

            if cv2.waitKey(10) & 0xFF == ord('r'): #pose로 수정필요 
                counter = 0
                lcounter = 0
                rcounter = 0
                stage = "Reset"
            if cv2.waitKey(10) & 0xFF == 27:
                menu_num=0
                break

    if (menu_num == 3) :
        print("Menu 3")
        cv2.imshow('MediaPipe Menu', image)

    if (menu_num == 4) :
        print("Menu 4")
        cv2.imshow('MediaPipe Menu', image)



############################################

#cap.release()
#cv2.destroyAllWindows()
#execute magic mirror

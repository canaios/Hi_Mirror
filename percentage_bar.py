import cv2
import mediapipe as mp
import numpy as np  #배열지원

mp_drawing = mp.solutions.drawing_utils #pose 그리기
mp_pose = mp.solutions.pose #pose 측정

cap = cv2.VideoCapture(0) # 실시간 동영상 캡쳐

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Curl counter variables
counter = 0
lcounter=0
rcounter =0
stage1 = None
stage2 = None
stage = None

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


## Setup mediapipe instance
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(rangle), 
                           tuple(np.multiply(relbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )


            if langle < rangle:
                per = np.interp(langle, (10,170),(100,0))
                bar = np.interp(langle, (10,170),(100,0))
            else :
                per = np.interp(rangle, (10,170),(100,0))
                bar = np.interp(rangle, (10,170),(100,0))
                

            
            # Curl counter logic
            if langle > 152:
                stage1 = "left down"
            if rangle > 152:
                stage2 = "right down"
            if (langle < 35 and stage1 =='left down'):
                stage1="left up"
                stage = "left up"
                lcounter +=1
                counter = lcounter + rcounter
                print(counter)
            if (rangle < 35 and stage2 =='right down'):
                stage2="right up"
                stage = "right up"
                rcounter +=1
                counter = lcounter + rcounter
                print(counter)
            if (langle > 130) and (rangle > 130):
                stage = "down"

        except:
            pass


                
        
        cv2.rectangle(image, (40,300), (70,400), (255,255,255), cv2.FILLED)
        cv2.rectangle(image, (40,400-int(bar)), (70,400), (170,145,116), cv2.FILLED)
        cv2.rectangle(image, (40,300), (70, 400), (80,80,80), 3)
        cv2.putText(image, f'{int(per)}%', (30,280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,180,180), 3, cv2.LINE_AA)




        
        # Render curl counter
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
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('r'):
            counter = 0
            lcounter = 0
            rcounter = 0
            stage = "Reset"
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

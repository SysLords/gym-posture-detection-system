import cv2
import mediapipe as mp
import numpy as np
import PoseModule as pm
import time

# Initialize webcam capture
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

# Variables for bicep curl counting
count = 0
stage = "down"  # Initialize stage to "down"
feedback = "Press 'r' for right arm or 'l' for left arm"
arm = None  # 'r' for right, 'l' for left

# Posture feedback variables
posture_feedback = ""
back_bend_tolerance = 15  # degrees of error allowed in back posture
speed_threshold = 1.5  # Minimum time for controlled curl (seconds)
last_curl_time = None

def count_curl(elbow_angle, hip_angle, current_stage, count, last_curl_time):
    global posture_feedback

    # Check posture
    posture_feedback = ""
    if hip_angle < 180 - back_bend_tolerance:
        posture_feedback = "Keep your back straight!"

    current_time = time.time()

    # Count the curl if posture is correct
    if posture_feedback == "":
        if current_stage == "down" and elbow_angle < 40:
            current_stage = "up"
            last_curl_time = current_time  # Start timing when the curl goes up
        elif current_stage == "up" and elbow_angle > 160:
            current_stage = "down"
            # Check if the curl was done too quickly after completing it
            if last_curl_time is not None and current_time - last_curl_time < speed_threshold:
                posture_feedback = "Slow down your curls!"
            count += 1  # Only increment the counter after completing the curl

    return current_stage, count, posture_feedback, last_curl_time

while cap.isOpened():
    ret, img = cap.read()

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0 and arm is not None:
        if arm == 'r':
            # Right arm angles
            elbow = detector.findAngle(img, 11, 13, 15)
            hip = detector.findAngle(img, 11, 23, 25)
        else:
            # Left arm angles
            elbow = detector.findAngle(img, 12, 14, 16)
            hip = detector.findAngle(img, 12, 24, 26)

        # Count curls and check posture
        stage, count, posture_feedback, last_curl_time = count_curl(elbow, hip, stage, count, last_curl_time)

        # Calculate percentage and bar progress for visualization
        per = np.interp(elbow, (30, 160), (0, 100))
        bar = np.interp(elbow, (30, 160), (380, 50))

        # Draw progress bar
        cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
        cv2.rectangle(img, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Draw curl counter
        cv2.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(count), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Draw feedback
        cv2.rectangle(img, (500, 0), (640, 40), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, feedback, (500, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Posture feedback
        if posture_feedback:
            cv2.putText(img, posture_feedback, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Draw which arm is being tracked
        arm_text = "Right Arm" if arm == 'r' else "Left Arm"
        cv2.putText(img, arm_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Bicep Curl Counter', img)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        arm = 'r'
        count = 0
        stage = "down"
        feedback = "Right arm selected"
        last_curl_time = None
    elif key == ord('l'):
        arm = 'l'
        count = 0
        stage = "down"
        feedback = "Left arm selected"
        last_curl_time = None

cap.release()
cv2.destroyAllWindows()
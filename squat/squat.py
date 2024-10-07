import sys
import mediapipe as mp
import cv2
import numpy as np


def findAngle(a, b, c, minVis=0.8):
    # Finds the angle at b with endpoints a and c
    # Returns -1 if below minimum visibility threshold
    # Takes lm_arr elements

    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])

        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba)
                                              * np.linalg.norm(bc))) * (180 / np.pi)

        if angle > 180:
            return 360 - angle
        else:
            return angle
    else:
        return -1


def legState(angle):
    if angle < 0:
        return 0  # Joint is not being picked up
    elif angle < 105:
        return 1  # Squat range
    elif angle < 150:
        return 2  # Transition range
    else:
        return 3  # Upright range


if __name__ == "__main__":

    # Init mediapipe drawing and pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize webcam video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    # Main Detection Loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # Initialize Reps and Body State
        repCount = 0
        lastState = 9

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read from webcam.")
                break

            # Resize frame for better visualization
            frame = cv2.resize(frame, (1024, 600))

            try:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False

                # Detect Pose Landmarks
                lm = pose.process(frame_rgb).pose_landmarks
                if not lm:
                    raise Exception("Pose not detected")

                lm_arr = lm.landmark

            except Exception as e:
                print("Please step into frame:", e)
                cv2.imshow("Squat Rep Counter", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Allow write, convert back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Calculate Angles
            rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])  # Right Hip-Knee-Ankle
            lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])  # Left Hip-Knee-Ankle

            # Calculate state
            rState = legState(rAngle)
            lState = legState(lAngle)
            state = rState * lState

            # Determine body state and rep count
            if state == 0:  # One or both legs not detected
                if rState == 0:
                    print("Right Leg Not Detected")
                if lState == 0:
                    print("Left Leg Not Detected")
            elif state % 2 == 0 or rState != lState:  # Transition phase
                if lastState == 1:
                    if lState == 2 or lState == 1:
                        print("Fully extend left leg")
                    if rState == 2 or lState == 1:
                        print("Fully extend right leg")
                else:
                    if lState == 2 or lState == 3:
                        print("Fully retract left leg")
                    if rState == 2 or lState == 3:
                        print("Fully retract right leg")
            else:  # Good form
                if state == 1 or state == 9:
                    if lastState != state:
                        lastState = state
                        if lastState == 1:
                            print("GOOD!")
                            repCount += 1

            print("Squats: " + str(repCount))

            cv2.imshow("Squat Rep Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

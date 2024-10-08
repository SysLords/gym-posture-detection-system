import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(A, B, C):
    A = np.array([A.x, A.y])
    B = np.array([B.x, B.y])
    C = np.array([C.x, C.y])

    AB = A - B
    BC = C - B

    cosine_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure cosine is in range to avoid math errors

    return np.degrees(angle)


angle_definitions = {
    'right_plank': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    'left_plank': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

start_time = None
holding_plank = False

def check_horizontal_alignment(shoulder, hip, threshold=0.1):
    """Checks if the vertical distance between shoulder and hip is within the threshold for a plank."""
    shoulder_y = shoulder.y
    hip_y = hip.y
    return abs(shoulder_y - hip_y) < threshold  # Threshold can be adjusted based on experimentation

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame from camera. Exiting...")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        side = 'right' if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility else 'left'

        if side == 'right':
            angle_name = 'right_plank'
        else:
            angle_name = 'left_plank'

        shoulder, hip, ankle = angle_definitions[angle_name]

        if (landmarks[shoulder].visibility > 0.5 and
            landmarks[hip].visibility > 0.5 and
            landmarks[ankle].visibility > 0.5):

            # Check if the user is horizontally aligned in a plank position
            if check_horizontal_alignment(landmarks[shoulder], landmarks[hip]):

                plank_angle = calculate_angle(
                    landmarks[shoulder],
                    landmarks[hip],
                    landmarks[ankle]
                )

                feedback = ''
                if plank_angle > 190:
                    feedback = 'Back is too high'
                    holding_plank = False
                elif plank_angle < 170:
                    feedback = 'Back is too low'
                    holding_plank = False
                else:
                    if not holding_plank:
                        holding_plank = True
                        start_time = time.time()

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if feedback:
                    cv2.putText(image, feedback,
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if holding_plank:
                    elapsed_time = time.time() - start_time
                    cv2.putText(image, f'Time: {int(elapsed_time)} sec',
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                feedback = "Not in plank position, adjust posture"
                cv2.putText(image, feedback,
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            feedback = "Make sure your entire body is visible"
            cv2.putText(image, feedback,
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Plank Form & Timer', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

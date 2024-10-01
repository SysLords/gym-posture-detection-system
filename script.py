import cv2
import mediapipe as mp
import numpy as np

# Function to calculate the angle between three points.
def calculate_angle(A, B, C):
    A = np.array([A.x, A.y])  # First point
    B = np.array([B.x, B.y])  # Middle point (joint)
    C = np.array([C.x, C.y])  # Last point

    AB = A - B  # Vector from A to B
    BC = C - B  # Vector from B to C

    # Calculate the cosine of the angle between vectors AB and BC
    cosine_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(cosine_angle)  # Angle in radians

    return np.degrees(angle)  # Convert to degrees

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera.
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame from camera. Exiting...")
        break

    # Convert the image to RGB and process it with MediaPipe Pose.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Convert the image back to BGR for display.
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Define key joint angles for both sides:
        # Right Side Angles
        right_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        )
        right_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        )
        right_hip_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        )

        # Left Side Angles
        left_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        )
        left_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        )
        left_hip_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        )

        # Display angles on the image, next to the respective joints:
        h, w, c = image.shape
        # Right Side
        cv2.putText(image, f'{int(right_elbow_angle)}', 
                    (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w), 
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'{int(right_knee_angle)}', 
                    (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), 
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'{int(right_hip_angle)}', 
                    (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), 
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Left Side
        cv2.putText(image, f'{int(left_elbow_angle)}', 
                    (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w), 
                     int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'{int(left_knee_angle)}', 
                    (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w), 
                     int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'{int(left_hip_angle)}', 
                    (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), 
                     int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image
    cv2.imshow('MediaPipe Pose', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
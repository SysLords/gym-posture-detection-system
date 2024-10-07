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

# Angle definitions for different body parts.
angle_definitions = {
    'right_elbow': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    'left_elbow': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    'right_shoulder': (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    'left_shoulder': (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    'neck': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER),
    'right_hip': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    'left_hip': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    'right_knee': (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    'left_knee': (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    'right_ankle': (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE, mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX),
    'left_ankle': (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE, mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX),
    'torso_bend': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    'hip_flexion': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    'right_arm_torso': (mp.solutions.pose.PoseLandmark.RIGHT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    'left_arm_torso': (mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    'right_leg_spread': (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    'left_leg_spread': (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
}

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
        h, w, _ = image.shape

        # Iterate through the angle definitions and calculate the angles.
        for angle_name, (point1, point2, point3) in angle_definitions.items():
            angle = calculate_angle(
                landmarks[point1], 
                landmarks[point2], 
                landmarks[point3]
            )

            # Display the calculated angle on the image at the middle joint position.
            joint_x = int(landmarks[point2].x * w)
            joint_y = int(landmarks[point2].y * h)
            cv2.putText(image, f'{angle_name}: {int(angle)}', 
                        (joint_x, joint_y), 
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

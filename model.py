import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from tqdm import tqdm  # For progress bar

def calculate_angle(A, B, C):
    A = np.array([A.x, A.y])
    B = np.array([B.x, B.y])
    C = np.array([C.x, C.y])
    AB = A - B
    BC = C - B
    cosine_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    angles = {}
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Define and calculate angles
        angle_definitions = {
            'right_elbow': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            'left_elbow': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            'right_shoulder': (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            'left_shoulder': (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            'neck': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
            'right_hip': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            'left_hip': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            'right_knee': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            'left_knee': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            'right_ankle': (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
            'left_ankle': (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            'torso_bend': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            'hip_flexion': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            'right_arm_torso': (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            'left_arm_torso': (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            'right_leg_spread': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE),
            'left_leg_spread': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
        }
        
        for angle_name, (p1, p2, p3) in angle_definitions.items():
            if (landmarks[p1].visibility > 0.5 and 
                landmarks[p2].visibility > 0.5 and 
                landmarks[p3].visibility > 0.5):
                angle = calculate_angle(landmarks[p1], landmarks[p2], landmarks[p3])
                angles[angle_name] = angle
    
    return angles

def process_exercises(root_folder):
    results = []
    for exercise_name in tqdm(os.listdir(root_folder), desc="Exercises"):  # Adding progress bar
        exercise_folder = os.path.join(root_folder, exercise_name)
        if os.path.isdir(exercise_folder):
            for image_name in tqdm(os.listdir(exercise_folder), desc=f"Processing {exercise_name}", leave=False):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(exercise_folder, image_name)
                    angles = process_image(image_path)
                    result = {'exercise': exercise_name, 'image': image_name, **angles}
                    results.append(result)
    return results

def save_to_csv(results, output_file):
    if not results:
        print("No results to save.")
        return
    
    # Use all possible keys from the results for the CSV header
    all_keys = set(k for result in results for k in result.keys())
    fieldnames = ['exercise', 'image'] + list(all_keys - {'exercise', 'image'})
    
    file_exists = os.path.isfile(output_file)
    
    # Append to the file if it exists
    with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header if file doesn't exist
        for result in results:
            writer.writerow(result)

# Main execution
root_folder = './archive'
output_file = 'exercise_angles.csv'

print("Starting exercise pose analysis...")
results = process_exercises(root_folder)
save_to_csv(results, output_file)

print(f"\nAnalysis complete. Results saved to {output_file}")

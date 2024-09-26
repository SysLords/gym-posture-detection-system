# Gym Posture Detection System Using QIDK
## Introduction

This project aims to create a real-time posture detection system using the Qualcomm Innovators Development Kit (QIDK) and a camera module. The system helps users improve their exercise form by analyzing body movements and providing corrective feedback. By enhancing posture, it helps reduce the risk of injury and improves workout efficiency.
Project Overview

---

# The project consists of:

- Pose estimation to detect body key points during workouts.
- Posture classification to identify correct and incorrect postures for various exercises.
- Feedback generation to help users improve their posture in real-time.

---

# Hardware

- Qualcomm Innovators Development Kit (QIDK) for processing posture data.
- Camera Module for real-time video capture.

# Software

- Pose Estimation Algorithm (e.g., MediaPipe) for detecting key body points.
- Machine Learning Model for posture classification.

---

## Deliverables

1. Deployed System: A real-time posture detection system with corrective feedback.
2. Metrics: Tracks accuracy of posture, common mistakes, and insights into injury prevention.

---

# Getting Started
## Prerequisites

1. Qualcomm Innovators Development Kit (QIDK)
2. Camera module
3. Python, OpenCV, and TensorFlow
4. MediaPipe for pose estimation
5. Kaggle dataset of workout exercises here

## Installation

1. Clone the repository:
   
`git clone https://github.com/your-username/gym-posture-detection-system.git
cd gym-posture-detection-system`

2. Install dependencies:

`pip install -r requirements.txt`

3. Set up QIDK and the camera module for real-time video capture.

## Usage

1. Pose Detection: Use the camera to capture live workout sessions.
2. Posture Analysis: The model will classify the exercise and detect any postural deviations.
3. Feedback: Real-time corrective feedback is displayed to help improve posture.

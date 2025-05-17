import streamlit as st
import cv2
from detector import PoseDetector
from pose_module import calculate_angle
import numpy as np

st.title("💪 AI Fitness & Pose Detection")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    rep_count = 0
    stage = None

    while run:
        ret, frame = cap.read()
        frame, results = detector.detect(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 160:
                stage = "down"
            if angle < 50 and stage == 'down':
                stage = "up"
                rep_count += 1

            cv2.putText(frame, f'Reps: {rep_count}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
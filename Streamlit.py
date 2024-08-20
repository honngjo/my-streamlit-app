import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import torch
import pickle
import os
import random
import tempfile

# Streamlit 페이지 설정
st.set_page_config(
    page_title="운동 분석",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이전 알림 시간 기록
previous_alert_time = 0
counter = 0
current_stage = ""
posture_status = [None]

def most_frequent(data):
    return max(data, key=data.count)

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

st.title("운동 자세 분석")

menu_selection = st.sidebar.selectbox("운동 선택", ("벤치프레스", "스쿼트", "데드리프트"))
counter_display = st.sidebar.empty()
counter_display.header(f"현재 카운터: {counter}회")

model_weights_path = {
    "벤치프레스": "./models/benchpress/benchpress.pkl",
    "스쿼트": "./models/squat/squat.pkl",
    "데드리프트": "./models/deadlift/deadlift.pkl"
}

model_e_path = model_weights_path.get(menu_selection, "./models/benchpress/benchpress.pkl")
with open(model_e_path, "rb") as f:
    model_e = pickle.load(f)

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    st.error("카메라를 열 수 없습니다. 올바른 장치가 연결되었는지 확인하세요.")
    st.stop()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=2,
)

confidence_threshold = st.sidebar.slider("관절점 추적 신뢰도 임계값", 0.0, 1.0, 0.7)

neck_angle_display = st.sidebar.empty()
left_shoulder_angle_display = st.sidebar.empty()
right_shoulder_angle_display = st.sidebar.empty()
left_elbow_angle_display = st.sidebar.empty()
right_elbow_angle_display = st.sidebar.empty()
left_hip_angle_display = st.sidebar.empty()
right_hip_angle_display = st.sidebar.empty()
left_knee_angle_display = st.sidebar.empty()
right_knee_angle_display = st.sidebar.empty()
left_ankle_angle_display = st.sidebar.empty()
right_ankle_angle_display = st.sidebar.empty()

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        st.error("카메라에서 프레임을 읽을 수 없습니다.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    results_pose = pose.process(frame)

    if results_pose.pose_landmarks is not None:
        landmarks = results_pose.pose_landmarks.landmark
        nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y]

        neck_angle = calculateAngle(left_shoulder, nose, left_hip)
        left_elbow_angle = calculateAngle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculateAngle(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = calculateAngle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculateAngle(right_elbow, right_shoulder, right_hip)
        left_hip_angle = calculateAngle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculateAngle(right_shoulder, right_hip, right_knee)
        left_knee_angle = calculateAngle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculateAngle(right_hip, right_knee, right_ankle)
        left_ankle_angle = calculateAngle(left_knee, left_ankle, left_heel)
        right_ankle_angle = calculateAngle(right_knee, right_ankle, right_heel)

        neck_angle_display.text(f"목 각도: {neck_angle:.2f}°")
        left_shoulder_angle_display.text(f"왼쪽 어깨 각도: {left_shoulder_angle:.2f}°")
        right_shoulder_angle_display.text(f"오른쪽 어깨 각도: {right_shoulder_angle:.2f}°")
        left_elbow_angle_display.text(f"왼쪽 팔꿈치 각도: {left_elbow_angle:.2f}°")
        right_elbow_angle_display.text(f"오른쪽 팔꿈치 각도: {right_elbow_angle:.2f}°")
        left_hip_angle_display.text(f"왼쪽 엉덩이 각도: {left_hip_angle:.2f}°")
        right_hip_angle_display.text(f"오른쪽 엉덩이 각도: {right_hip_angle:.2f}°")
        left_knee_angle_display.text(f"왼쪽 무릎 각도: {left_knee_angle:.2f}°")
        right_knee_angle_display.text(f"오른쪽 무릎 각도: {right_knee_angle:.2f}°")
        left_ankle_angle_display.text(f"왼쪽 발목 각도: {left_ankle_angle:.2f}°")
        right_ankle_angle_display.text(f"오른쪽 발목 각도: {right_ankle_angle:.2f}°")

        try:
            row = [coord for res in results_pose.pose_landmarks.landmark for coord in [res.x, res.y, res.z, res.visibility]]
            X = pd.DataFrame([row])
            exercise_class = model_e.predict(X)[0]
            exercise_class_prob = model_e.predict_proba(X)[0]

            if "down" in exercise_class:
                current_stage = "down"
                posture_status.append(exercise_class)
            elif current_stage == "down" and "up" in exercise_class:
                current_stage = "up"
                counter += 1
                posture_status.append(exercise_class)
                counter_display.header(f"진행횟수: {counter}회")
                
                if "correct" not in most_frequent(posture_status):
                    current_time = time.time()
                    if current_time - previous_alert_time >= 3:
                        now = datetime.datetime.now()
                        feedback_messages = {
                            "excessive_arch": [
                                ("허리를 너무 많이 굽히지 말고 자연스러운 자세를 유지하세요.", "./resources/sounds/excessive_arch_1.mp3"),
                                ("골반을 약간 들어 올리고 복근을 조여 등을 평평하게 유지합니다.", "./resources/sounds/excessive_arch_2.mp3")
                            ],
                            "arms_spread": [
                                ("그립이 너무 넓습니다. 바를 조금 더 좁게 잡으세요.", "./resources/sounds/arms_spread_1.mp3"),
                                ("바를 잡을 때는 어깨 너비보다 약간 넓게 잡습니다.", "./resources/sounds/arms_spread_2.mp3")
                            ],
                            "spine_neutral": [
                                ("척추가 과도하게 휘어지지 않도록 주의하세요.", "./resources/sounds/spine_neutral_feedback_1.mp3"),
                                ("가슴을 들어 올리고 어깨를 뒤로 젖힙니다.", "./resources/sounds/spine_neutral_feedback_2.mp3")
                            ],
                            "caved_in_knees": [
                                ("스쿼트를 하는 동안 무릎이 구부러지지 않도록 주의하세요.", "./resources/sounds/caved_in_knees_feedback_1.mp3"),
                                ("엉덩이를 뒤로 밀어서 무릎과 발가락이 일직선이 되도록 합니다.", "./resources/sounds/caved_in_knees_feedback_2.mp3")
                            ]
                        }

                        frequent_posture = most_frequent(posture_status)
                        if frequent_posture in feedback_messages:
                            selected_message, selected_music = random.choice(feedback_messages[frequent_posture])
                            st.error(selected_message)
                            st.audio(selected_music)
                            posture_status = []
                            previous_alert_time = current_time
                        else:
                            posture_status = []
                elif "correct" in most_frequent(posture_status):
                    st.audio("./resources/sounds/correct.mp3")
                    st.info("올바른 자세로 운동을 하고 있습니다.")
                    posture_status = []
        except Exception as e:
            st.error(f"Error: {e}")

        for landmark in mp_pose.PoseLandmark:
            if landmarks[landmark.value].visibility >= confidence_threshold:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )

    FRAME_WINDOW.image(frame)

camera.release()
pose.close()

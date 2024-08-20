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
import tempfile  # 추가된 부분

st.set_page_config(
    page_title="test",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
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

# 각도 계산 함수
def calculateAngle(a, b, c):
    a = np.array(a)  # 첫 번째 지점
    b = np.array(b)  # 중간 지점
    c = np.array(c)  # 끝 지점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Streamlit 앱 초기화
st.title("test")

# Sidebar에 메뉴 추가
menu_selection = st.selectbox("운동 선택", ("벤치프레스", "스쿼트", "데드리프트"))
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
    camera.release()
    
# 임시 디렉터리를 사용하도록 설정 (추가된 부분)
temp_dir = tempfile.gettempdir()

# Mediapipe Pose 모델 초기화: 최소 감지 신뢰도=0.5, 최소 추적 신뢰도=0.7, 모델 복잡도=2를 준다.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=2,
)

# 신뢰도 임계값 슬라이더
confidence_threshold = st.sidebar.slider("관절점 추적 신뢰도 임계값", 0.0, 1.0, 0.7)

# 각도 표시를 위한 빈 영역 초기화
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

while True:
    ret, frame = camera.read()
    
    # 카메라에서 프레임을 제대로 읽었는지 확인
    if not ret:
        st.error("카메라에서 프레임을 읽을 수 없습니다.")
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)  # 프레임 좌우반전

    # Pose estimation을 수행
    results_pose = pose.process(frame)

    # Pose 결과 처리
    if results_pose.pose_landmarks is not None:
        landmarks = results_pose.pose_landmarks.landmark
        nose = [
            landmarks[mp_pose.PoseLandmark.NOSE].x,
            landmarks[mp_pose.PoseLandmark.NOSE].y,
        ]
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
        ]
        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
        ]
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
        ]
        left_heel = [
            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
        ]
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
        ]
        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
        ]
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
        ]
        right_heel = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,
        ]

        # 각도 계산
        neck_angle = (
            calculateAngle(left_shoulder, nose, left_hip)
            + calculateAngle(right_shoulder, nose, right_hip) / 2
        )
        left_elbow_angle = calculateAngle(
            left_shoulder, left_elbow, left_wrist
        )
        right_elbow_angle = calculateAngle(
            right_shoulder, right_elbow, right_wrist
        )
        left_shoulder_angle = calculateAngle(
            left_elbow, left_shoulder, left_hip
        )
        right_shoulder_angle = calculateAngle(
            right_elbow, right_shoulder, right_hip
        )
        left_hip_angle = calculateAngle(
            left_shoulder, left_hip, left_knee
        )
        right_hip_angle = calculateAngle(
            right_shoulder, right_hip, right_knee
        )
        left_knee_angle = calculateAngle(
            left_hip, left_knee, left_ankle
        )
        right_knee_angle = calculateAngle(
            right_hip, right_knee, right_ankle
        )
        left_ankle_angle = calculateAngle(
            left_knee, left_ankle, left_heel
        )
        right_ankle_angle = calculateAngle(
            right_knee, right_ankle, right_heel
        )

        # 각도 표시 업데이트
        neck_angle_display.text(f"목 각도: {neck_angle:.2f}°")
        left_shoulder_angle_display.text(
            f"왼쪽 어깨 각도: {left_shoulder_angle:.2f}°"
        )
        right_shoulder_angle_display.text(
            f"오른쪽 어깨 각도: {right_shoulder_angle:.2f}°"
        )
        left_elbow_angle_display.text(
            f"왼쪽 어깨 각도: {left_elbow_angle:.2f}°"
        )
        right_elbow_angle_display.text(
            f"오른쪽 어깨 각도: {right_elbow_angle:.2f}°"
        )
        left_hip_angle_display.text(
            f"왼쪽 엉덩이 각도: {left_hip_angle:.2f}°"
        )
        right_hip_angle_display.text(
            f"오른쪽 엉덩이 각도: {right_hip_angle:.2f}°"
        )
        left_knee_angle_display.text(
            f"왼쪽 무릎 각도: {left_knee_angle:.2f}°"
        )
        right_knee_angle_display.text(
            f"오른쪽 무릎 각도: {right_knee_angle:.2f}°"
        )
        left_ankle_angle_display.text(
            f"왼쪽 발목 각도: {left_ankle_angle:.2f}°"
        )
        right_ankle_angle_display.text(
            f"오른쪽 발목 각도: {right_ankle_angle:.2f}°"
        )

        # 자세 인식 모델에 대한 예측
        try:
            row = [
                coord
                for res in results_pose.pose_landmarks.landmark
                for coord in [res.x, res.y, res.z, res.visibility]
            ]
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
                        if "excessive_arch" in most_frequent(posture_status):
                            options = [
                                (
                                    "허리를 너무 많이 굽히지 말고 자연스러운 자세를 유지하세요.",
                                    "./resources/sounds/excessive_arch_1.mp3",
                                ),
                                (
                                    "골반을 약간 들어 올리고 복근을 조여 등을 평평하게 유지합니다.",
                                    "./resources/sounds/excessive_arch_2.mp3",
                                ),
                            ]
                            selected_option = random.choice(options)
                            selected_message = selected_option[0]
                            selected_music = selected_option[1]
                            st.error(selected_message)
                            st.audio(selected_music)
                            posture_status = []
                            previous_alert_time = current_time
                        elif "arms_spread" in most_frequent(posture_status):
                            options = [
                                (
                                    "그립이 너무 넓습니다. 바를 조금 더 좁게 잡으세요. ",
                                    "./resources/sounds/arms_spread_1.mp3",
                                ),
                                (
                                    "바를 잡을 때는 어깨 너비보다 약간 넓게 잡습니다.",
                                    "./resources/sounds/arms_spread_2.mp3",
                                ),
                            ]
                            selected_option = random.choice(options)
                            selected_message = selected_option[0]
                            selected_music = selected_option[1]
                            st.error(selected_message)
                            st.audio(selected_music)
                            posture_status = []
                            previous_alert_time = current_time
                        elif "spine_neutral" in most_frequent(posture_status):
                            options = [
                                (
                                    "척추가 과도하게 휘어지지 않도록 주의하세요.",
                                    "./resources/sounds/spine_neutral_feedback_1.mp3",
                                ),
                                (
                                    "가슴을 들어 올리고 어깨를 뒤로 젖힙니다.",
                                    "./resources/sounds/spine_neutral_feedback_2.mp3",
                                ),
                            ]
                            selected_option = random.choice(options)
                            selected_message = selected_option[0]
                            selected_music = selected_option[1]
                            st.error(selected_message)
                            st.audio(selected_music)
                            posture_status = []
                            previous_alert_time = current_time
                        elif "caved_in_knees" in most_frequent(posture_status):
                            options = [
                                (
                                    "스쿼트를 하는 동안 무릎이 구부러지지 않도록 주의하세요.",
                                    "./resources/sounds/caved_in_knees_feedback_1.mp3",
                                ),
                                (
                                    "엉덩이를 뒤로 밀어서 무릎과 발가락이 일직선이 되도록 합니다.",
                                    "./resources/sounds/caved_in_knees_feedback_2.mp3",
                                ),
                            ]
                            selected_option = random.choice(options)
                            selected_message = selected_option[0]
                            selected_music = selected_option[1]
                            st.error(selected_message)
                            st.audio(selected_music)
                            posture_status = []
                            previous_alert_time = current_time
                        elif "feet_spread" in most_frequent(posture_status):
                            st.error(
                                "어깨 너비 정도로 자세를 좁힙니다."
                            )
                            st.audio(
                                "./resources/sounds/feet_spread.mp3"
                            )
                            posture_status = []
                            previous_alert_time = current_time
                        elif "arms_narrow" in most_frequent(posture_status):
                            st.error(
                                "그립이 너무 넓습니다. 바를 조금 더 좁게 잡으세요."
                            )
                            st.audio(
                                "./resources/sounds/arms_narrow.mp3"
                            )
                            posture_status = []
                            previous_alert_time = current_time
                elif "correct" in most_frequent(posture_status):
                    st.audio("./resources/sounds/correct.mp3")
                    st.info(
                        "올바른 자세로 운동을 하고 있습니다."
                    )
                    posture_status = []
        except Exception as e:
            st.error(f"Error: {e}")

        # 랜드마크 그리기
        for landmark in mp_pose.PoseLandmark:
            if landmarks[landmark.value].visibility >= confidence_threshold:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )

    # 원본 프레임을 출력
    FRAME_WINDOW.image(frame)

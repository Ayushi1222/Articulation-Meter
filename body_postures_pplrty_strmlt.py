import streamlit as st
import pandas as pd
import numpy as np
from yt_dlp import YoutubeDL
import mediapipe as mp
import cv2
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
model_path = '/home/rudra-gupta/Desktop/Articulation-Meter/-Articulation-Meter-/Project Files/linear_regression_model.pkl'
model = joblib.load(model_path)

# Streamlit app
st.title("YouTube Video View Predictor")

# Input field for YouTube URL
youtube_url = st.text_input("Enter YouTube URL:")

if youtube_url:
    # Download video from YouTube
    with st.spinner("Downloading video..."):
        ydl_opts = {'format': 'best', 'outtmpl': 'downloaded_video.%(ext)s'}
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_file = ydl.prepare_filename(info_dict)
            likes = info_dict.get('like_count', 0)  # Extract likes from metadata

    # Process video using MediaPipe to extract features
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    shoulder_midpoints = []
    head_turn_angles_left = []
    head_turn_angles_right = []
    left_hand_distances = []
    right_hand_distances = []

    cap = cv2.VideoCapture(video_file)
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    with st.spinner("Processing video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_model.process(frame_rgb)

            if results.pose_landmarks:
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_midpoints.append(shoulder_midpoint_x)

                left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
                right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                eye_line_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
                eye_left_nose_vector = np.array([nose.x - left_eye.x, nose.y - left_eye.y])
                eye_right_nose_vector = np.array([right_eye.x - nose.x, right_eye.y - nose.y])

                dot_product_left = np.dot(eye_line_vector, eye_left_nose_vector)
                eye_line_magnitude = np.linalg.norm(eye_line_vector)
                eye_left_nose_magnitude = np.linalg.norm(eye_left_nose_vector)

                dot_product_right = np.dot(eye_line_vector, eye_right_nose_vector)
                eye_right_nose_magnitude = np.linalg.norm(eye_right_nose_vector)

                cosine_angle_left = dot_product_left / (eye_line_magnitude * eye_left_nose_magnitude)
                cosine_angle_right = dot_product_right / (eye_line_magnitude * eye_right_nose_magnitude)

                head_turn_angle_left = np.arccos(np.clip(cosine_angle_left, -1.0, 1.0)) * (180 / np.pi)
                head_turn_angle_right = np.arccos(np.clip(cosine_angle_right, -1.0, 1.0)) * (180 / np.pi)

                head_turn_angles_left.append(head_turn_angle_left)
                head_turn_angles_right.append(head_turn_angle_right)

            if results.left_hand_landmarks:
                left_hand_landmarks = results.left_hand_landmarks.landmark
                left_hand_distances += [np.sqrt(lm.x ** 2 + lm.y ** 2 + lm.z ** 2) for lm in left_hand_landmarks]
            
            if results.right_hand_landmarks:
                right_hand_landmarks = results.right_hand_landmarks.landmark
                right_hand_distances += [np.sqrt(lm.x ** 2 + lm.y ** 2 + lm.z ** 2) for lm in right_hand_landmarks]

    cap.release()

    # Calculate features for prediction
    shoulder_midpoints_mode = np.median(shoulder_midpoints) if shoulder_midpoints else np.nan
    head_turn_angles_mean = np.mean(head_turn_angles_left + head_turn_angles_right) if head_turn_angles_left or head_turn_angles_right else np.nan
    left_hand_mode = np.median(left_hand_distances) if left_hand_distances else np.nan
    right_hand_median = np.median(right_hand_distances) if right_hand_distances else np.nan
    shoulder_head_interaction = shoulder_midpoints_mode * head_turn_angles_mean if not np.isnan(shoulder_midpoints_mode) and not np.isnan(head_turn_angles_mean) else np.nan
    left_right_hand_sum = left_hand_mode + right_hand_median if not np.isnan(left_hand_mode) and not np.isnan(right_hand_median) else np.nan

    # Create a DataFrame for the model
    data = {
        'likes': [likes],
        'shoulder_midpoints_mode': [shoulder_midpoints_mode],
        'head_turn_angles_mean': [head_turn_angles_mean],
        'left_hand_mode': [left_hand_mode],
        'right_hand_median': [right_hand_median],
        'shoulder_head_interaction': [shoulder_head_interaction],
        'left_right_hand_sum': [left_right_hand_sum]
    }
    df = pd.DataFrame(data)

    # Check for missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    if missing_columns:
        st.warning(f"Warning: Missing values in columns: {', '.join(missing_columns)}")
        st.info("Prediction may be less accurate due to missing data.")

    # Impute missing values with median
    df_imputed = df.fillna(df.median())

    # Make prediction
    try:
        predicted_views = model.predict(df_imputed)[0]
        st.success(f"Predicted Views: {int(predicted_views)}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("Please ensure the model is trained with the same features as provided in the input data.")

    # Optional: Correlation analysis or visualization
    if st.checkbox("Show correlation matrix"):
        corr_matrix = df_imputed.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
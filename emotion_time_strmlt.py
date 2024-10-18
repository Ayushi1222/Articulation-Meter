import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from moviepy.editor import VideoFileClip
from yt_dlp import YoutubeDL
import os

# Load pre-trained emotion detection model
emotion_model = load_model('/home/rudra-gupta/Desktop/Articulation-Meter/-Articulation-Meter-/Project Files/emotion_model.h5')
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to process video
def process_video(video_url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'downloaded_video.%(ext)s',
    }

    # Download the video
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_file = ydl.prepare_filename(info_dict)

    # Load the video using moviepy
    clip = VideoFileClip(video_file)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_file)

    # Initialize variables
    frame_count = 0
    emotions = []

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame_grey)

        # Detect emotions in the frame and append to the emotions list
        if faces is not None:
            for (x, y, w, h) in faces:
                roi_gray = frame_grey[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0) / 255.0
                predictions = emotion_model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                detected_emotion = emotion_labels[max_index]
                emotions.append(detected_emotion)

        # Break loop after processing 10 seconds (fps * 10)
        if frame_count >= clip.fps * 10:
            break

    cap.release()
    return emotions

# Streamlit app
st.title("Emotion Detection from Video")

# Input for YouTube video URL
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Process Video"):
    if video_url:
        with st.spinner("Processing video..."):
            emotions = process_video(video_url)
            st.success("Video processed!")

            # Display results
            st.subheader("Detected Emotions (in 10-second intervals):")
            st.write(emotions)

            # Plot the emotions as a graph
            plt.figure(figsize=(10, 5))
            plt.plot(emotions, marker='o')
            plt.xticks(range(len(emotions)), [f"Frame {i*10}" for i in range(len(emotions))])
            plt.ylabel('Emotions')
            plt.title('Detected Emotions Over Time')
            plt.grid()
            plt.show()
            st.pyplot(plt)
    else:
        st.error("Please enter a valid YouTube video URL.")


# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from moviepy.editor import VideoFileClip
# from yt_dlp import YoutubeDL
# import tempfile
# import os

# # Initialize MediaPipe solutions
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# mp_holistic = mp.solutions.holistic

# # Load models
# face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# model_path = '/home/rudra-gupta/Desktop/Articulation-Meter/-Articulation-Meter-/Project Files/emotion_model.h5'

# # Try loading the model and print appropriate messages
# try:
#     emotion_model = load_model(model_path)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")

# # Define emotion labels
# emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

# def process_video(youtube_url):
#     # Download the video using yt-dlp
#     with tempfile.TemporaryDirectory() as temp_dir:
#         ydl_opts = {
#             'format': 'best',
#             'outtmpl': os.path.join(temp_dir, 'downloaded_video.%(ext)s'),
#         }

#         with YoutubeDL(ydl_opts) as ydl:
#             info_dict = ydl.extract_info(youtube_url, download=True)
#             video_file = ydl.prepare_filename(info_dict)

#         # Process the video
#         clip = VideoFileClip(video_file)
#         cap = cv2.VideoCapture(video_file)

#         # Get video properties
#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
#         frame_count = int(cap.get(7))
#         fps = int(cap.get(5))
#         video_length_seconds = frame_count / fps

#         # Initialize MediaPipe models
#         pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#         # Initialize lists for data storage
#         shoulder_midpoints = []
#         head_turn_angles_right = []
#         head_turn_angles_left = []
#         emotions = []
#         left_hand = []
#         right_hand = []

#         # Set up Streamlit progress bar
#         progress_bar = st.progress(0)
#         frame_placeholder = st.empty()
#         graph_placeholder = st.empty()

#         frame_index = 0
#         frame_counter = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_counter += 1
#             if frame_counter >= fps/6:
#                 frame_counter = 0
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 faces = face_haar_cascade.detectMultiScale(frame_grey)

#                 # Process emotions
#                 try:
#                     for (x, y, w, h) in faces:
#                         cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
#                         roi_gray = frame_grey[y - 5:y + h + 5, x - 5:x + w + 5]
#                         roi_gray = cv2.resize(roi_gray, (48, 48))
#                         image_pixels = img_to_array(roi_gray)
#                         image_pixels = np.expand_dims(image_pixels, axis=0)
#                         image_pixels /= 255
#                         predictions = emotion_model.predict(image_pixels)
#                         max_index = np.argmax(predictions[0])
#                         detected_emotion = emotion_labels[max_index]
#                         emotions.append(detected_emotion)
#                 except:
#                     emotions.append(None)

#                 # Process pose
#                 results = pose.process(frame_rgb)
#                 if results.pose_landmarks is not None:
#                     left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#                     right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

#                     if left_shoulder and right_shoulder:
#                         shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
#                         shoulder_midpoints.append(shoulder_midpoint_x)
#                     else:
#                         shoulder_midpoints.append(None)

#                     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#                     # Process head turn angles
#                     left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
#                     right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]
#                     nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

#                     if left_eye and right_eye and nose:
#                         eye_line_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
#                         eye_left_nose_vector = np.array([nose.x - left_eye.x, nose.y - left_eye.y])
#                         eye_right_nose_vector = np.array([right_eye.x - nose.x, right_eye.y - nose.y])

#                         dot_product_left = np.dot(eye_line_vector, eye_left_nose_vector)
#                         eye_line_magnitude = np.linalg.norm(eye_line_vector)
#                         eye_left_nose_magnitude = np.linalg.norm(eye_left_nose_vector)

#                         dot_product_right = np.dot(eye_line_vector, eye_right_nose_vector)
#                         eye_right_nose_magnitude = np.linalg.norm(eye_right_nose_vector)

#                         cosine_angle_left = dot_product_left / (eye_line_magnitude * eye_left_nose_magnitude)
#                         cosine_angle_right = dot_product_right / (eye_line_magnitude * eye_right_nose_magnitude)

#                         head_turn_angle_left = np.arccos(cosine_angle_left) * (180 / np.pi)
#                         head_turn_angle_right = np.arccos(cosine_angle_right) * (180 / np.pi)

#                         head_turn_angles_left.append(head_turn_angle_left)
#                         head_turn_angles_right.append(head_turn_angle_right)
#                     else:
#                         head_turn_angles_left.append(None)
#                         head_turn_angles_right.append(None)
#                 else:
#                     shoulder_midpoints.append(None)
#                     head_turn_angles_left.append(None)
#                     head_turn_angles_right.append(None)

#                 # Process hand movements
#                 hand_results = holistic_model.process(frame_rgb)

#                 left_hand_sum_distance = 0
#                 if hand_results.left_hand_landmarks is not None:
#                     mp_drawing.draw_landmarks(frame, hand_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#                     left_hand_landmarks = hand_results.left_hand_landmarks.landmark
#                     for landmark in left_hand_landmarks:
#                         distance = np.sqrt(landmark.x**2 + landmark.y**2 + landmark.z**2)
#                         left_hand_sum_distance += distance
#                     left_hand.append(left_hand_sum_distance)
#                 else:
#                     left_hand.append(None)

#                 right_hand_sum_distance = 0
#                 if hand_results.right_hand_landmarks is not None:
#                     mp_drawing.draw_landmarks(frame, hand_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#                     right_hand_landmarks = hand_results.right_hand_landmarks.landmark
#                     for landmark in right_hand_landmarks:
#                         distance = np.sqrt(landmark.x**2 + landmark.y**2 + landmark.z**2)
#                         right_hand_sum_distance += distance
#                     right_hand.append(right_hand_sum_distance)
#                 else:
#                     right_hand.append(None)

#                 frame_index += 1

#                 # Update Streamlit components
#                 progress = int((frame_index / frame_count) * 100)
#                 progress_bar.progress(progress)

#                 frame_placeholder.image(frame, channels="BGR", use_column_width=True)

#                 # Create and update graphs
#                 fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

#                 # Shoulder Movement Graph
#                 ax1.plot(np.arange(frame_index), shoulder_midpoints, color='red', linewidth=1, marker='o', markersize=1)
#                 ax1.set_ylabel('Shoulder Midpoint')
#                 ax1.set_xlim([0, video_length_seconds * 6])

#                 # Head Turn Angle Graph
#                 ax2.plot(np.arange(frame_index), head_turn_angles_left, color='red', linewidth=1, marker='o', markersize=1, label='Left Turn Angle')
#                 ax2.plot(np.arange(frame_index), head_turn_angles_right, color='blue', linewidth=1, marker='o', markersize=1, label='Right Turn Angle')
#                 ax2.set_ylabel('Head Turn Angle (degrees)')
#                 ax2.set_xlim([0, video_length_seconds * 6])
#                 ax2.legend()

#                 # Emotion Graph
#                 x_values = np.arange(len(emotions))
#                 ax3.scatter(x_values, [emotion_labels.index(e) if e is not None else -1 for e in emotions], color='green', s=5)
#                 ax3.set_ylim(-1, len(emotion_labels))
#                 ax3.set_xlim(0, video_length_seconds * 6)
#                 ax3.set_yticks(range(len(emotion_labels)))
#                 ax3.set_yticklabels(emotion_labels)
#                 ax3.set_ylabel('Emotion')
#                 ax3.grid(True, linestyle='--', alpha=0.6)

#                 # Hand Movement Graph
#                 ax4.plot(np.arange(frame_index), left_hand, color='red', linewidth=1, marker='o', markersize=1, label='Left hand movement')
#                 ax4.plot(np.arange(frame_index), right_hand, color='blue', linewidth=1, marker='o', markersize=1, label='Right hand movement')
#                 ax4.set_ylabel('Hand movements')
#                 ax4.set_xlim([0, video_length_seconds * 6])
#                 ax4.legend()

#                 plt.tight_layout()
#                 graph_placeholder.pyplot(fig)
#                 plt.close(fig)

#         cap.release()

# def main():
#     st.markdown(
#         """
#         <style>
#         @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap');

#         /* Main container */
#         .stApp {
#             background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#             font-family: 'Roboto', sans-serif;
#         }

#         /* Entrance */
#         .entrance {
#             text-align: center;
#             padding: 2rem;
#             background: rgba(255, 255, 255, 0.1);
#             border-radius: 15px;
#             backdrop-filter: blur(10px);
#             margin-bottom: 2rem;
#         }

#         .entrance h1 {
#             font-size: 3.5rem;
#             font-weight: 700;
#             color: #2c3e50;
#             margin-bottom: 1rem;
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#         }

#         .entrance .emoji {
#             font-size: 3rem;
#             margin: 0 0.5rem;
#         }

#         /* Video container */
#         .video-container {
#             position: relative;
#             overflow: hidden;
#             border-radius: 15px;
#             box-shadow: 0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23);
#         }

#         .video-container::before,
#         .video-container::after {
#             content: '';
#             position: absolute;
#             top: 0;
#             width: 50px;
#             height: 100%;
#             background: linear-gradient(to right, rgba(255,255,255,0.2), transparent);
#             animation: shine 2s infinite;
#         }

#         .video-container::before {
#             left: -50px;
#         }

#         .video-container::after {
#             right: -50px;
#             transform: rotateY(180deg);
#         }

#         @keyframes shine {
#             0% { transform: translateX(-100%); }
#             100% { transform: translateX(100%); }
#         }

#         /* Graphs container */
#         .graphs-container {
#             display: flex;
#             justify-content: space-between;
#             flex-wrap: wrap;
#             margin-top: 2rem;
#         }

#         .graph-item {
#             width: calc(25% - 1rem);
#             background-color: rgba(255, 255, 255, 0.8);
#             border-radius: 15px;
#             padding: 1rem;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             transition: all 0.3s ease;
#         }

#         .graph-item:hover {
#             transform: translateY(-5px);
#             box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
#         }

#         .metric {
#             text-align: center;
#             font-size: 1.2rem;
#             font-weight: 500;
#             margin-top: 1rem;
#         }

#         /* Rest of the styles remain the same */
#         ...

#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     st.markdown("""
#     <div class="entrance">
#         <h1>Video Analysis App <span class="emoji">🎥</span><span class="emoji">📊</span></h1>
#     </div>
#     """, unsafe_allow_html=True)
    
#     youtube_url = st.text_input("Enter YouTube URL:")
    
#     if st.button("Analyze Video"):
#         if youtube_url:
#             process_video(youtube_url)
#         else:
#             st.warning("Please enter a valid YouTube URL.")

# if __name__ == "__main__":
#     main()




import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from moviepy.editor import VideoFileClip
from yt_dlp import YoutubeDL
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def create_and_update_graphs(frame_index, shoulder_midpoints, head_turn_angles_left, head_turn_angles_right, emotions, left_hand, right_hand, video_length_seconds, emotion_labels):
    # Create four separate figures
    figs = []

    # Shoulder Movement Graph
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=np.arange(frame_index), y=shoulder_midpoints, mode='lines+markers',
                              line=dict(color='red', width=2), marker=dict(size=3)))
    fig1.update_layout(title='Shoulder Movement', xaxis_title='Frame', yaxis_title='Shoulder Midpoint',
                       height=400, width=600)
    fig1.update_xaxes(range=[0, video_length_seconds * 6])
    figs.append(fig1)

    # Head Turn Angle Graph
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.arange(frame_index), y=head_turn_angles_left, mode='lines+markers',
                              name='Left Turn Angle', line=dict(color='red', width=2), marker=dict(size=3)))
    fig2.add_trace(go.Scatter(x=np.arange(frame_index), y=head_turn_angles_right, mode='lines+markers',
                              name='Right Turn Angle', line=dict(color='blue', width=2), marker=dict(size=3)))
    fig2.update_layout(title='Head Turn Angle', xaxis_title='Frame', yaxis_title='Angle (degrees)',
                       height=400, width=600, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig2.update_xaxes(range=[0, video_length_seconds * 6])
    figs.append(fig2)

    # Emotion Graph
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=np.arange(len(emotions)), 
                              y=[emotion_labels.index(e) if e is not None else -1 for e in emotions],
                              mode='markers', marker=dict(color='green', size=5)))
    fig3.update_layout(title='Emotion', xaxis_title='Frame', yaxis_title='Emotion',
                       height=400, width=600)
    fig3.update_yaxes(ticktext=emotion_labels, tickvals=list(range(len(emotion_labels))))
    fig3.update_xaxes(range=[0, video_length_seconds * 6])
    figs.append(fig3)

    # Hand Movement Graph
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=np.arange(frame_index), y=left_hand, mode='lines+markers',
                              name='Left hand movement', line=dict(color='red', width=2), marker=dict(size=3)))
    fig4.add_trace(go.Scatter(x=np.arange(frame_index), y=right_hand, mode='lines+markers',
                              name='Right hand movement', line=dict(color='blue', width=2), marker=dict(size=3)))
    fig4.update_layout(title='Hand Movement', xaxis_title='Frame', yaxis_title='Movement',
                       height=400, width=600, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig4.update_xaxes(range=[0, video_length_seconds * 6])
    figs.append(fig4)
# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Load models
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model_path = '/home/rudra-gupta/Desktop/Articulation-Meter/-Articulation-Meter-/Project Files/emotion_model.h5'

# Try loading the model and print appropriate messages
try:
    emotion_model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']


def process_video(youtube_url):
    # Download the video using yt-dlp
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(temp_dir, 'downloaded_video.%(ext)s'),
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_file = ydl.prepare_filename(info_dict)

        # Process the video
        clip = VideoFileClip(video_file)
        cap = cv2.VideoCapture(video_file)

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_count = int(cap.get(7))
        fps = int(cap.get(5))
        video_length_seconds = frame_count / fps

        # Initialize MediaPipe models
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Initialize lists for data storage
        shoulder_midpoints = []
        head_turn_angles_right = []
        head_turn_angles_left = []
        emotions = []
        left_hand = []
        right_hand = []

        # Set up Streamlit progress bar
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        graph_placeholder = st.empty()

        frame_index = 0
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter >= fps/6:
                frame_counter = 0
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_haar_cascade.detectMultiScale(frame_grey)

                # Process emotions
                try:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
                        roi_gray = frame_grey[y - 5:y + h + 5, x - 5:x + w + 5]
                        roi_gray = cv2.resize(roi_gray, (48, 48))
                        image_pixels = img_to_array(roi_gray)
                        image_pixels = np.expand_dims(image_pixels, axis=0)
                        image_pixels /= 255
                        predictions = emotion_model.predict(image_pixels)
                        max_index = np.argmax(predictions[0])
                        detected_emotion = emotion_labels[max_index]
                        emotions.append(detected_emotion)
                except:
                    emotions.append(None)

                # Process pose
                results = pose.process(frame_rgb)
                if results.pose_landmarks is not None:
                    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                    if left_shoulder and right_shoulder:
                        shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
                        shoulder_midpoints.append(shoulder_midpoint_x)
                    else:
                        shoulder_midpoints.append(None)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Process head turn angles
                    left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
                    right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]
                    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                    if left_eye and right_eye and nose:
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

                        head_turn_angle_left = np.arccos(cosine_angle_left) * (180 / np.pi)
                        head_turn_angle_right = np.arccos(cosine_angle_right) * (180 / np.pi)

                        head_turn_angles_left.append(head_turn_angle_left)
                        head_turn_angles_right.append(head_turn_angle_right)
                    else:
                        head_turn_angles_left.append(None)
                        head_turn_angles_right.append(None)
                else:
                    shoulder_midpoints.append(None)
                    head_turn_angles_left.append(None)
                    head_turn_angles_right.append(None)

                # Process hand movements
                hand_results = holistic_model.process(frame_rgb)

                left_hand_sum_distance = 0
                if hand_results.left_hand_landmarks is not None:
                    mp_drawing.draw_landmarks(frame, hand_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    left_hand_landmarks = hand_results.left_hand_landmarks.landmark
                    for landmark in left_hand_landmarks:
                        distance = np.sqrt(landmark.x**2 + landmark.y**2 + landmark.z**2)
                        left_hand_sum_distance += distance
                    left_hand.append(left_hand_sum_distance)
                else:
                    left_hand.append(None)

                right_hand_sum_distance = 0
                if hand_results.right_hand_landmarks is not None:
                    mp_drawing.draw_landmarks(frame, hand_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    right_hand_landmarks = hand_results.right_hand_landmarks.landmark
                    for landmark in right_hand_landmarks:
                        distance = np.sqrt(landmark.x**2 + landmark.y**2 + landmark.z**2)
                        right_hand_sum_distance += distance
                    right_hand.append(right_hand_sum_distance)
                else:
                    right_hand.append(None)

                frame_index += 1

                # Update Streamlit components
                progress = int((frame_index / frame_count) * 100)
                progress_bar.progress(progress)

                frame_placeholder.image(frame, channels="BGR", use_column_width=True)

                # Create and update graphs
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
                fig.subplots_adjust(hspace=0.4, wspace=0.4)
                # Shoulder Movement Graph
                ax1.plot(np.arange(frame_index), shoulder_midpoints, color='red', linewidth=1, marker='o', markersize=1)
                ax1.set_ylabel('Shoulder Midpoint')
                ax1.set_xlim([0, video_length_seconds * 6])

                # Head Turn Angle Graph
                ax2.plot(np.arange(frame_index), head_turn_angles_left, color='red', linewidth=1, marker='o', markersize=1, label='Left Turn Angle')
                ax2.plot(np.arange(frame_index), head_turn_angles_right, color='blue', linewidth=1, marker='o', markersize=1, label='Right Turn Angle')
                ax2.set_ylabel('Head Turn Angle (degrees)')
                ax2.set_xlim([0, video_length_seconds * 6])
                ax2.legend()

                # Emotion Graph
                x_values = np.arange(len(emotions))
                ax3.scatter(x_values, [emotion_labels.index(e) if e is not None else -1 for e in emotions], color='green', s=5)
                ax3.set_ylim(-1, len(emotion_labels))
                ax3.set_xlim(0, video_length_seconds * 6)
                ax3.set_yticks(range(len(emotion_labels)))
                ax3.set_yticklabels(emotion_labels)
                ax3.set_ylabel('Emotion')
                ax3.grid(True, linestyle='--', alpha=0.6)

                # Hand Movement Graph
                ax4.plot(np.arange(frame_index), left_hand, color='red', linewidth=1, marker='o', markersize=1, label='Left hand movement')
                ax4.plot(np.arange(frame_index), right_hand, color='blue', linewidth=1, marker='o', markersize=1, label='Right hand movement')
                ax4.set_ylabel('Hand movements')
                ax4.set_xlim([0, video_length_seconds * 6])
                ax4.legend()

                plt.tight_layout()
                graph_placeholder.pyplot(fig)
                plt.close(fig)

        cap.release()

def main():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap');

    /* Main container */
    .stApp {
   background: linear-gradient(135deg, #5a9bd4, #a2c2e0, #f5f7fa, #c3cfe2);

    background-size: 400% 400%; /* For animation effect */
    animation: gradient-animation 15s ease infinite; /* Animation for smooth transition */
    font-family: 'Roboto', sans-serif;
}
@keyframes gradient-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

    /* Entrance */
    .entrance {
        text-align: center;
        padding: 3rem;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        margin-bottom: 3rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease-in-out;
    }

    .entrance:hover {
        transform: scale(1.05);
    }

    .entrance h1 {
        font-size: 4rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .entrance .emoji {
        font-size: 3rem;
        margin: 0 0.5rem;
    }

    /* Gradient Animated Button */
    .animated-button {
        background: linear-gradient(270deg, #ff6b6b, #f7b42c, #18debb, #6a11cb, #f7b42c);
        background-size: 800% 800%;
        color: white;
        padding: 15px 30px;
        font-size: 1.2rem;
        font-weight: 500;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        transition: background-position 0.5s ease-in-out;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    }

    .animated-button:hover {
        background-position: 100% 0;
        box-shadow: 0px 15px 20px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }

    /* Form Input Fields */
    .form-input {
        padding: 10px 15px;
        width: 100%;
        border: 2px solid rgba(44, 62, 80, 0.1);
        border-radius: 10px;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        transition: border-color 0.3s ease;
    }

    .form-input:focus {
        border-color: #6a11cb;
        outline: none;
    }

    /* Customizing the form container */
    .form-container {
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 1rem;
        color: #7f8c8d;
    }

    .footer a {
        color: #6a11cb;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

.stTextInput {
            border: 2px solid #5a9bd4; 
            border-radius: 8px; 
            padding: 10px 15px; 
            font-size: 1.2rem; 
            width: 100%; 
            max-width: 500px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
            transition: border-color 0.3s, box-shadow 0.3s; 
        }

        .stTextInput:focus {
            border-color: #007bff; 
            box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); 
        }

        /* Style for the button */
.stButton {
    # background: linear-gradient(135deg, #007bff, #5a9bd4); /* Gradient background */
    border: none; /* Remove default border */
    border-radius: 8px; /* Rounded corners */
    color: white; /* Text color */
    font-size: 1.2rem; /* Font size */
    padding: 12px 20px; /* Padding for button */
    cursor: pointer; /* Pointer cursor on hover */
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Transition for hover effects */
}

.stButton:hover {
    transform: scale(1.05); /* Slightly enlarge on hover */
    box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3); /* Shadow effect on hover */
}

.stButton:focus {
    outline: none; /* Remove outline for focused state */
    box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); /* Shadow effect on focus */
}

        

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="entrance">
        <h1>Speaker Articulation Analysis Simulator<span class="emoji">🎥</span><span class="emoji">📊</span></h1>
    </div>
    """, unsafe_allow_html=True)
    
    youtube_url = st.text_input("Enter YouTube URL:")
    
    if st.button("Analyze Video"):
        if youtube_url:
            process_video(youtube_url)
        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()



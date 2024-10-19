import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import time
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header

# Initialize MediaPipe Pose and Holistic
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize lists to store data for graphs
left_hand = []
right_hand = []
shoulder_midpoints = []
head_turn_angles = []
engagement_level = []

# Set the graph display parameters
graph_height = 200
line_width = 1

# Set video dimensions
output_width = 800
output_height = 600

# Create a video capture object
cap = cv2.VideoCapture(0)

# Create an output video writer
output_video = './output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 20, (output_width, output_height))

# Initialize frame index
frame_index = 0

def calculate_engagement(left_hand, right_hand, shoulder_midpoints, head_turn_angles):
    # Use recent history to determine engagement level
    history_len = 10  # Number of frames to consider for engagement calculation

    # Slicing the recent history
    recent_left_hand = left_hand[-history_len:]
    recent_right_hand = right_hand[-history_len:]
    recent_shoulder_midpoints = shoulder_midpoints[-history_len:]
    recent_head_turn_angles = head_turn_angles[-history_len:]

    # Calculate average movements
    avg_left_hand = np.nanmean([x for x in recent_left_hand if x is not None])
    avg_right_hand = np.nanmean([x for x in recent_right_hand if x is not None])
    avg_shoulder = np.nanmean([x for x in recent_shoulder_midpoints if x is not None])
    avg_head_turn = np.nanmean([x for x in recent_head_turn_angles if x is not None])

    # Engagement calculation based on movement thresholds
    hand_threshold = 0.1  # Movement threshold for hands
    shoulder_threshold = 0.05  # Movement threshold for shoulders
    head_turn_threshold = 5.0  # Angle threshold for head turns (degrees)

    left_hand_movement = avg_left_hand is not None and avg_left_hand > hand_threshold
    right_hand_movement = avg_right_hand is not None and avg_right_hand > hand_threshold
    shoulder_movement = avg_shoulder is not None and avg_shoulder > shoulder_threshold
    head_movement = avg_head_turn is not None and avg_head_turn > head_turn_threshold

    if left_hand_movement and right_hand_movement:
        return 'inspired'
    elif left_hand_movement or right_hand_movement:
        return 'interactive'
    else:
        return 'attentive'

def process_frame(frame, frame_index):
    global left_hand, right_hand, shoulder_midpoints, head_turn_angles, engagement_level

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(frame_grey)

    # Detect pose in the frame
    results = pose.process(frame_rgb)
    frame_rgb.flags.writeable = False
    hand_results = holistic_model.process(frame_rgb)
    frame_rgb.flags.writeable = True

    # Draw landmarks and calculate distances for left and right hands
    left_hand_sum_distance = 0
    right_hand_sum_distance = 0

    if hand_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        left_hand_landmarks = hand_results.left_hand_landmarks.landmark
        for landmark in left_hand_landmarks:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            distance = np.sqrt(x**2 + y**2 + z**2)
            left_hand_sum_distance += distance
        left_hand.append(left_hand_sum_distance)
    else:
        left_hand.append(None)

    if hand_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        right_hand_landmarks = hand_results.right_hand_landmarks.landmark
        for landmark in right_hand_landmarks:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            distance = np.sqrt(x**2 + y**2 + z**2)
            right_hand_sum_distance += distance
        right_hand.append(right_hand_sum_distance)
    else:
        right_hand.append(None)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_midpoint = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_midpoints.append(shoulder_midpoint)

        left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Calculate head turn angle
        if left_eye and right_eye and nose:
            eye_line_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
            nose_vector = np.array([nose.x - (left_eye.x + right_eye.x) / 2, nose.y - (left_eye.y + right_eye.y) / 2])
            cosine_angle = np.dot(eye_line_vector, nose_vector) / (np.linalg.norm(eye_line_vector) * np.linalg.norm(nose_vector))
            head_turn_angle = np.arccos(cosine_angle) * (180 / np.pi)
            head_turn_angles.append(head_turn_angle)
        else:
            head_turn_angles.append(None)
    else:
        shoulder_midpoints.append(None)
        head_turn_angles.append(None)

    # Calculate engagement level
    engagement = calculate_engagement(left_hand, right_hand, shoulder_midpoints, head_turn_angles)
    engagement_level.append(engagement)

    # Filter out None values for graphs
    filtered_left_hand = [point for point in left_hand if point is not None]
    filtered_right_hand = [point for point in right_hand if point is not None]
    filtered_shoulder_midpoints = [point for point in shoulder_midpoints if point is not None]
    filtered_head_turn_angles = [point for point in head_turn_angles if point is not None]
    filtered_engagement_level = [e for e in engagement_level if e is not None]

    # Calculate min and max values for y-axis limits
    min_left_hand = min(filtered_left_hand) if filtered_left_hand else 0
    max_left_hand = max(filtered_left_hand) if filtered_left_hand else 1
    min_right_hand = min(filtered_right_hand) if filtered_right_hand else 0
    max_right_hand = max(filtered_right_hand) if filtered_right_hand else 1
    min_shoulder_midpoint = min(filtered_shoulder_midpoints) if filtered_shoulder_midpoints else 0
    max_shoulder_midpoint = max(filtered_shoulder_midpoints) if filtered_shoulder_midpoints else 1
    min_head_turn_angle = min(filtered_head_turn_angles) if filtered_head_turn_angles else 0
    max_head_turn_angle = max(filtered_head_turn_angles) if filtered_head_turn_angles else 1
    min_engagement = 1
    max_engagement = 3

    # Create separate figures for each graph
    fig, axes = plt.subplots(4, 1, figsize=(output_width / 100, 4 * graph_height / 100))  # Adjust graph size

    # Hand Movement Graph
    if len(filtered_left_hand) > 0 and len(filtered_right_hand) > 0:
        axes[0].plot(filtered_left_hand, color='green', linewidth=line_width, marker='o', markersize=1, label='Left hand movement')
        axes[0].plot(filtered_right_hand, color='yellow', linewidth=line_width, marker='o', markersize=1, label='Right hand movement')
    axes[0].set_ylim([min(min_left_hand, min_right_hand), max(max_left_hand, max_right_hand)])
    axes[0].set_xlim([0, frame_index])
    axes[0].axis('on')
    axes[0].set_ylabel('Hand Movements', fontsize=6)
    axes[0].legend(loc='upper right', fontsize=5)

    # Shoulder Midpoints Graph
    if len(filtered_shoulder_midpoints) > 0:
        axes[1].plot(filtered_shoulder_midpoints, color='red', linewidth=line_width, marker='o', markersize=1, label='Shoulder midpoint')
    axes[1].set_ylim([min_shoulder_midpoint, max_shoulder_midpoint])
    axes[1].set_xlim([0, frame_index])
    axes[1].axis('on')
    axes[1].set_ylabel('Shoulder Midpoint', fontsize=6)
    axes[1].legend(loc='upper right', fontsize=5)

    # Head Turn Angle Graph
    if len(filtered_head_turn_angles) > 0:
        axes[2].plot(filtered_head_turn_angles, color='purple', linewidth=line_width, marker='o', markersize=1, label='Head turn angle')
    axes[2].set_ylim([min_head_turn_angle, max_head_turn_angle])
    axes[2].set_xlim([0, frame_index])
    axes[2].axis('on')
    axes[2].set_ylabel('Head Turn Angle', fontsize=6)
    axes[2].legend(loc='upper right', fontsize=5)

    # Engagement Level Graph
    if len(filtered_engagement_level) > 0:
        engagement_colors = {'attentive': 'red', 'interactive': 'yellow', 'inspired': 'green'}
        engagement_numeric = [1 if e == 'attentive' else 2 if e == 'interactive' else 3 for e in filtered_engagement_level]
        colors = [engagement_colors[e] for e in filtered_engagement_level]
        axes[3].scatter(range(len(filtered_engagement_level)), engagement_numeric, c=colors, s=5, label='Engagement level')
    axes[3].set_ylim([min_engagement, max_engagement])
    axes[3].set_xlim([0, frame_index])
    axes[3].set_yticks([1, 2, 3])
    axes[3].set_yticklabels(['Attentive', 'Interactive', 'Inspired'])
    axes[3].axis('on')
    axes[3].set_ylabel('Engagement Level', fontsize=6)
    axes[3].legend(loc='upper right', fontsize=5)

    plt.tight_layout()

    return frame, fig, engagement

def get_engagement_feedback(engagement_level):
    feedback = {
        'attentive': {
            'color': 'red',
            'message': """
            💡 Engagement Feedback: You seem attentive but could be more interactive!
            
            Tips to improve:
            - Try using more hand gestures while speaking
            - Vary your body posture occasionally
            - Engage with head movements to emphasize points
            - Consider incorporating more dynamic movements
            """
        },
        'interactive': {
            'color': 'yellow',
            'message': """
            🌟 Engagement Feedback: Great interaction level!
            
            Keep improving by:
            - Adding more emphasis to key points with hand gestures
            - Maintaining this energy level
            - Try incorporating different gesture patterns
            - Consider adding more facial expressions
            """
        },
        'inspired': {
            'color': 'green',
            'message': """
            ⭐ Engagement Feedback: Excellent articulation!
            
            To maintain this level:
            - Keep up the dynamic movements
            - Continue using varied gestures
            - Your energy is perfect - maintain it!
            - You're doing great with body language
            """
        }
    }
    return feedback[engagement_level]


def reset_data():
    """Reset all data lists and counters"""
    global left_hand, right_hand, shoulder_midpoints, head_turn_angles, engagement_level
    left_hand = []
    right_hand = []
    shoulder_midpoints = []
    head_turn_angles = []
    engagement_level = []


def main():
    st.set_page_config(page_title="Hello Speaker!", layout="wide", initial_sidebar_state="collapsed")
    
    # Set light theme
    st.markdown("""
        <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #000000;
        }
        h1, h2, h3 {
            color: #1f77b4;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
        }
        .feedback-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Hello Speaker!")

    # Create three columns
    col1, col2, col3 = st.columns([2, 2, 1])

    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = True
    if 'frame_index' not in st.session_state:
        st.session_state.frame_index = 0

    # Control buttons column
    with col1:
        st.header("Live Camera Feed")
        video_placeholder = st.empty()
        
        # Create a row for buttons
        button_col1, button_col2 = st.columns(2)
        
        # Camera control button
        with button_col1:
            if st.button('Stop Camera' if st.session_state.camera_running else 'Start Camera'):
                st.session_state.camera_running = not st.session_state.camera_running
                if not st.session_state.camera_running:
                    st.success("Camera stopped! Click again to restart.")
                else:
                    st.success("Camera started!")
        
        # Refresh button
        with button_col2:
            if st.button('Refresh Graphs'):
                reset_data()
                st.session_state.frame_index = 0
                st.success("Graphs reset!")

    with col2:
        st.header("Engagement Analysis")
        graphs_placeholder = st.empty()

    with col3:
        st.header("Real-time Feedback")
        feedback_placeholder = st.empty()

    # Create a ThreadPoolExecutor for processing frames
    with ThreadPoolExecutor(max_workers=2) as executor:
        cap = cv2.VideoCapture(0)

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            # Submit frame for processing
            future = executor.submit(process_frame, frame, st.session_state.frame_index)
            frame, fig, engagement = future.result()

            # Display the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with col1:
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Display the graphs
            with col2:
                graphs_placeholder.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory

            # Display feedback
            with col3:
                feedback = get_engagement_feedback(engagement)
                feedback_placeholder.markdown(
                    f"""
                    <div style="background-color: {feedback['color']}20; padding: 20px; border-radius: 10px; border: 2px solid {feedback['color']}">
                    {feedback['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Update the Streamlit border color based on engagement level
            border_color = {'attentive': 'red', 'interactive': 'yellow', 'inspired': 'green'}[engagement]
            st.markdown(
                f"""
                <style>
                .stApp {{
                    border: 5px solid {border_color};
                    padding: 10px;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            st.session_state.frame_index += 1
            time.sleep(0.1)  # Adjust this to control frame rate

        cap.release()

        # Show summary when camera is stopped
        if not st.session_state.camera_running:
            st.markdown("### Session Summary")
            if len(engagement_level) > 0:  # Only show summary if there's data
                engagement_counts = {
                    'attentive': engagement_level.count('attentive'),
                    'interactive': engagement_level.count('interactive'),
                    'inspired': engagement_level.count('inspired')
                }
                
                total_frames = len(engagement_level)
                st.write("Engagement Distribution:")
                for level, count in engagement_counts.items():
                    percentage = (count / total_frames) * 100
                    st.progress(percentage / 100)
                    st.write(f"{level.capitalize()}: {percentage:.1f}%")

if __name__ == "__main__":
    main()

# run command : streamlit run "/home/rudra/Documents/GitHub/Articulation-Meter/Project Files/engagement_streamlit.py"
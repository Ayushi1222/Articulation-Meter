# Articulation Meter 🎙️📊

## Revolutionizing Public Speaking with AI-Powered Engagement Analysis

## 🚀 Project Overview

Articulation Meter is a groundbreaking project that leverages the power of artificial intelligence and multimodal data analysis to revolutionize public speaking and content creation. Inspired by the ancient concept of Navarasa, we've developed a cutting-edge system that analyzes various aspects of a speaker's performance to enhance engagement and predict popularity.

### 🌟 Key Features

- **Multimodal Analysis**: Combines body language, voice modulation, and facial expressions
- **Real-time Feedback**: Provides instant insights for speakers to improve their performance
- **Engagement Prediction**: Forecasts viewer engagement and popularity metrics
- **Comparative Analysis**: Benchmark against top influencers and speakers
- **Interactive Practice Room**: A virtual environment for honing public speaking skills

## 🧠 The Science Behind Articulation Meter

Our project draws inspiration from the ancient Indian concept of Navarasa, which describes nine emotions that form the foundation of effective communication and storytelling. By integrating this timeless wisdom with cutting-edge AI and machine learning techniques, we've created a tool that not only analyzes but also enhances the art of public speaking.

### 📊 Data Points Analyzed

1. **Body Posture**
   - Hand movements
   - Head angles
   - Shoulder movements

2. **Voice Analysis**
   - Pitch
   - Tone
   - Modulation
   - Pace

3. **Facial Emotions**
   - Micro-expressions
   - Emotional transitions

4. **Content Analysis**
   - Keyword density
   - Narrative structure
   - Rhetorical devices

## 💻 Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MediaPipe
- **Audio Processing**: Librosa
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly

## 🛠️ Installation and Setup

To get started with Articulation Meter, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TECHAMID/ArticulationMeter.git
   cd ArticulationMeter
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained models:
   ```bash
   python download_models.py
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run engagement_streamlit.py
   ```

## 📁 Project Structure

```
ArticulationMeter/
│
├── data/
│   ├── ted_data.xlsx
│   ├── train.txt
│   └── transcript.txt
│
├── models/
│   ├── emotion_model.h5
│   ├── linear_regression_model.pkl
│   └── model_pipeline.pkl
│
├── notebooks/
│   ├── Audio_Extraction.ipynb
│   ├── Model_Trained_View_Prediction.ipynb
│   ├── RealTimePlots_new.ipynb
│   ├── audio_transition_matrix.ipynb
│   ├── coorelation_pplrty__sm_hta_lh_rh.ipynb
│   └── video_features_popularity_analysis.ipynb
│
├── src/
│   ├── audio_processing.py
│   ├── body_posture_analysis.py
│   ├── emotion_detection.py
│   ├── engagement_metrics.py
│   └── video_processing.py
│
├── streamlit_apps/
│   ├── RealTimePlots_new_strmlt.py
│   ├── body_postures_pplrty_strmlt.py
│   ├── emotion_time_strmlt.py
│   └── engagement_streamlit.py
│
├── tests/
│   └── test_main_functions.py
│
├── utils/
│   ├── data_preprocessing.py
│   ├── visualization.py
│   └── youtube_downloader.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🚀 Features in Detail

### 1. Multimodal Analysis

Our system performs a comprehensive analysis of the speaker's performance by combining:

- **Body Language**: Tracking hand movements, head angles, and shoulder positions to assess confidence and emphasis.
- **Voice Modulation**: Analyzing pitch, tone, and pace to evaluate vocal variety and emotional conveyance.
- **Facial Expressions**: Detecting micro-expressions and emotional transitions to gauge authenticity and audience connection.

### 2. Real-time Feedback

Speakers receive instant, color-coded feedback on their performance:

- 🟢 Green: High engagement, continue with current style
- 🟡 Yellow: Moderate engagement, suggestions for improvement provided
- 🟥 Red: Low engagement, immediate adjustments recommended

### 3. Engagement Prediction

Utilizing machine learning models trained on extensive datasets, including TED Talks and popular YouTube content, to forecast:

- Viewer retention rates
- Like-to-view ratios
- Potential viral spread

### 4. Comparative Analysis

Benchmark your performance against top influencers and speakers:

- Analyze speaking styles of figures like PM Narendra Modi or Ranveer Allahbadia
- Identify key factors contributing to their engagement success
- Receive personalized recommendations to elevate your speaking style

### 5. Interactive Practice Room

A virtual environment designed for perfecting your public speaking skills:

- Real-time visual feedback through color-coded "mirrors"
- Categorization of your engagement style (Attentive, Interactive, Inspired)
- Personalized exercises to improve weak areas

## 🔮 Future Implications

1. **Content Recommendation Engines**: Enhance social media algorithms by incorporating engagement level predictions.
2. **Trend Forecasting**: Analyze current engagement trends to predict upcoming content themes and styles.
3. **Server Load Prediction**: Estimate potential server loads for hosting platforms based on predicted view counts and engagement levels.
4. **Educational Tools**: Integrate into public speaking courses and corporate training programs.
5. **Mental Health Applications**: Adapt the emotion detection system for mood tracking and mental health monitoring.

## 🤝 Contributing

We welcome contributions from the community! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- The TED Talks organization for providing invaluable data
- The open-source community for the various libraries and tools used in this project

<p align="center">Made with ❤️ by TECHAMID</p>

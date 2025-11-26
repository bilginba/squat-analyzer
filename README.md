# AI Video Squat Analysis

An AI-powered Python application to analyze squat exercises from video input using pose detection.

## Features

-   Real-time pose detection using MediaPipe
-   Squat form analysis (knee angle, hip angle, depth)
-   Repetition counting
-   Performance visualization with charts
-   Detailed analysis reports

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Web App (Recommended)

Run the interactive web application:

```bash
streamlit run app.py
```

Or use the provided batch file (Windows):

```bash
run_streamlit.bat
```

The app will open in your browser at `http://localhost:8501`

### Command Line

```bash
python main.py --video path/to/your/squat_video.mp4
```

## Output

-   Analysis charts (angles over time, depth analysis)
-   Squat count and form metrics
-   Visual overlay on video frames

## Requirements

-   Python 3.10 (for Streamlit Cloud deployment)
-   Python 3.8+ (for local development)
-   Webcam or video file of squat exercises

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Deploy from your repository
4. **Important**: Set Python version to **3.10** in Settings → Advanced settings
5. Add your `GOOGLE_API_KEY` in Secrets management

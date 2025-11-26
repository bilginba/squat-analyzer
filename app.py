"""
Modern Streamlit UI for AI-Powered Squat Analysis Application
"""
import streamlit as st
import cv2
import os
import tempfile
import time
import google.generativeai as genai
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pose_detector import PoseDetector
from squat_analyzer import SquatAnalyzer
from visualizer import SquatVisualizer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Squat Analyzer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'llm_analysis' not in st.session_state:
    st.session_state.llm_analysis = None


def configure_gemini(api_key):
    """Configure Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False


def get_llm_analysis(stats, data, api_key):
    """Get AI-powered analysis using Google Gemini"""
    try:
        if not configure_gemini(api_key):
            return None
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare data for analysis
        prompt = f"""
You are an expert fitness coach analyzing squat exercise performance. Based on the following data, provide a comprehensive analysis:

**Performance Metrics:**
- Total Squats Completed: {stats['total_squats']}
- Average Knee Angle: {stats['avg_knee_angle']:.1f}°
- Minimum Knee Angle: {stats['min_knee_angle']:.1f}°
- Maximum Knee Angle: {stats['max_knee_angle']:.1f}°
- Average Hip Angle: {stats['avg_hip_angle']:.1f}°
- Maximum Depth Achieved: {stats['max_depth']:.1f}%
- Total Frames Analyzed: {stats['total_frames']}

**Context:**
- Ideal squat depth: Knee angle below 90°
- Full range of motion: Knee angle from ~70° (bottom) to ~170° (standing)
- Proper form indicators: Consistent depth, controlled movement, good hip hinge

Please provide:
1. **Overall Performance Assessment** (2-3 sentences)
2. **Form Analysis** - Evaluate depth consistency and range of motion
3. **Strengths** - What the user is doing well (bullet points)
4. **Areas for Improvement** - Specific recommendations (bullet points)
5. **Next Steps** - Actionable advice for better squat performance

Keep the tone encouraging and professional. Format the response in clear sections.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error getting LLM analysis: {str(e)}")
        return None


def process_video(video_path, depth_threshold, min_knee_angle, progress_bar, status_text):
    """Process video and analyze squats"""
    # Initialize components
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    analyzer = SquatAnalyzer(depth_threshold=depth_threshold, min_knee_angle=min_knee_angle)
    visualizer = SquatVisualizer(output_dir="output")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"Could not open video file")
        return None, None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer
    output_path = os.path.join("output", "analyzed_video.mp4")
    os.makedirs("output", exist_ok=True)
    
    # Use H.264 codec for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} - Squats detected: {analyzer.squat_count}")
            
            # Detect pose
            landmarks, _ = detector.detect_pose(frame)
            
            # Analyze squat
            analysis = analyzer.analyze_frame(landmarks, frame_width, frame_height, detector)
            
            # Draw pose landmarks
            frame = detector.draw_landmarks(frame, landmarks)
            
            # Add analysis overlay
            if analysis:
                cv2.putText(frame, f"Squats: {analysis['squat_count']}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                state_color = (0, 255, 255) if analysis['squat_state'] == "down" else (255, 255, 255)
                cv2.putText(frame, f"State: {analysis['squat_state'].upper()}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
                
                if analysis['knee_angle']:
                    cv2.putText(frame, f"Knee: {analysis['knee_angle']:.1f}deg", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    if analysis['knee_coord']:
                        cv2.circle(frame, analysis['knee_coord'], 8, (0, 255, 255), -1)
                
                if analysis['hip_angle']:
                    cv2.putText(frame, f"Hip: {analysis['hip_angle']:.1f}deg", 
                               (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                feedback = analyzer.get_form_feedback(analysis['knee_angle'], analysis['depth_percent'])
                cv2.putText(frame, feedback, 
                           (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame
            output_writer.write(frame)
    
    finally:
        cap.release()
        output_writer.release()
        detector.close()
    
    # Get results
    stats = analyzer.get_summary_statistics()
    data = analyzer.get_analysis_data()
    
    # Store output video path
    st.session_state.video_path = output_path
    
    return stats, data


def create_plotly_charts(data, stats, fps=30):
    """Create interactive Plotly charts"""
    charts = {}
    
    # Angle timeline chart
    if data["knee_angles"] and data["hip_angles"]:
        time_seconds = np.array(data["timestamps"][:min(len(data["knee_angles"]), len(data["hip_angles"]))]) / fps
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_seconds[:len(data["knee_angles"])],
            y=data["knee_angles"],
            mode='lines',
            name='Knee Angle',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=time_seconds[:len(data["hip_angles"])],
            y=data["hip_angles"],
            mode='lines',
            name='Hip Angle',
            line=dict(color='#764ba2', width=3)
        ))
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Depth Threshold (90°)")
        
        fig.update_layout(
            title="Joint Angles Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Angle (degrees)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        charts['angles'] = fig
    
    # Depth analysis
    if data["depths"]:
        time_seconds = np.array(data["timestamps"][:len(data["depths"])]) / fps
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_seconds,
            y=data["depths"],
            mode='lines',
            name='Squat Depth',
            fill='tozeroy',
            line=dict(color='purple', width=3)
        ))
        
        fig.update_layout(
            title="Squat Depth Analysis",
            xaxis_title="Time (seconds)",
            yaxis_title="Depth (%)",
            template='plotly_white',
            height=400
        )
        charts['depth'] = fig
    
    # Distribution histogram
    if data["knee_angles"]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data["knee_angles"],
            nbinsx=30,
            name='Knee Angle Distribution',
            marker_color='#667eea'
        ))
        fig.add_vline(x=stats['avg_knee_angle'], line_dash="dash", line_color="red",
                     annotation_text=f"Avg: {stats['avg_knee_angle']:.1f}°")
        
        fig.update_layout(
            title="Knee Angle Distribution",
            xaxis_title="Angle (degrees)",
            yaxis_title="Frequency",
            template='plotly_white',
            height=400
        )
        charts['distribution'] = fig
    
    return charts


def main():
    # Header
    st.markdown('<h1 class="main-header">🏋️ AI-Powered Squat Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/squat.png", width=100)
        st.title("⚙️ Configuration")
        
        # Google API Key - Load from .env or allow manual input
        st.subheader("🤖 AI Analysis")
        env_api_key = os.getenv("GOOGLE_API_KEY")
        
        if env_api_key:
            api_key = env_api_key
            st.success("✅ API Key loaded from .env file")
        else:
            api_key = st.text_input("Google API Key", type="password", 
                                    help="Enter your Google Gemini API key for AI-powered analysis")
            
            if api_key:
                st.success("✅ API Key configured")
            else:
                st.info("ℹ️ Add your Google API key to .env file or enter it here")
        
        st.markdown("---")
        
        # Analysis parameters
        st.subheader("📊 Analysis Parameters")
        depth_threshold = st.slider("Depth Threshold (degrees)", 60, 120, 90,
                                    help="Knee angle threshold for valid squat")
        min_knee_angle = st.slider("Minimum Knee Angle (degrees)", 50, 90, 70,
                                   help="Minimum knee angle for bottom position")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This application uses AI to analyze your squat form:
        - **Pose Detection**: MediaPipe for real-time tracking
        - **Form Analysis**: Automated squat counting and metrics
        - **AI Insights**: Google Gemini for personalized feedback
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "🤖 AI Insights"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Your Squat Video")
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_video_path = tmp_file.name
                
                st.video(uploaded_file)
                
                if st.button("🚀 Analyze Video", key="analyze_btn"):
                    with st.spinner("Processing video..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        stats, data = process_video(
                            tmp_video_path,
                            depth_threshold,
                            min_knee_angle,
                            progress_bar,
                            status_text
                        )
                        
                        if stats and data:
                            st.session_state.stats = stats
                            st.session_state.analysis_data = data
                            st.session_state.processed = True
                            
                            # Get LLM analysis if API key provided
                            if api_key:
                                with st.spinner("Getting AI insights..."):
                                    llm_analysis = get_llm_analysis(stats, data, api_key)
                                    st.session_state.llm_analysis = llm_analysis
                            
                            status_text.empty()
                            progress_bar.empty()
                            st.balloons()
                            st.success("✅ Analysis complete! Check the Results Dashboard and AI Insights tabs.")
        
        with col2:
            st.subheader("ℹ️ Tips for Best Results")
            st.markdown("""
            - 📹 Use a side-view angle
            - 💡 Ensure good lighting
            - 👤 Keep full body in frame
            - 🎥 Stable camera position
            - ⏱️ Clear, controlled movements
            """)
    
    with tab2:
        if st.session_state.processed and st.session_state.stats:
            stats = st.session_state.stats
            data = st.session_state.analysis_data
            
            # Metrics row
            st.subheader("📈 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Squats</div>
                    <div class="metric-value">{stats['total_squats']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Knee Angle</div>
                    <div class="metric-value">{stats['avg_knee_angle']:.1f}°</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Min Knee Angle</div>
                    <div class="metric-value">{stats['min_knee_angle']:.1f}°</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Max Depth</div>
                    <div class="metric-value">{stats['max_depth']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Charts
            charts = create_plotly_charts(data, stats)
            
            if 'angles' in charts:
                st.plotly_chart(charts['angles'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if 'depth' in charts:
                    st.plotly_chart(charts['depth'], use_container_width=True)
            with col2:
                if 'distribution' in charts:
                    st.plotly_chart(charts['distribution'], use_container_width=True)
            
            # Detailed statistics table
            st.subheader("📋 Detailed Statistics")
            stats_df = pd.DataFrame({
                'Metric': [
                    'Total Squats',
                    'Average Knee Angle',
                    'Minimum Knee Angle',
                    'Maximum Knee Angle',
                    'Average Hip Angle',
                    'Average Depth',
                    'Maximum Depth',
                    'Total Frames'
                ],
                'Value': [
                    stats['total_squats'],
                    f"{stats['avg_knee_angle']:.2f}°",
                    f"{stats['min_knee_angle']:.2f}°",
                    f"{stats['max_knee_angle']:.2f}°",
                    f"{stats['avg_hip_angle']:.2f}°",
                    f"{stats['avg_depth']:.2f}%",
                    f"{stats['max_depth']:.2f}%",
                    stats['total_frames']
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Display and download analyzed video
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.markdown("---")
                st.subheader("🎥 Analyzed Video")
                st.video(st.session_state.video_path)
                
                st.markdown("---")
                st.subheader("📥 Download Results")
                with open(st.session_state.video_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Analyzed Video",
                        data=f,
                        file_name=f"squat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
        else:
            st.info("👆 Upload and analyze a video first to see results here.")
    
    with tab3:
        if st.session_state.llm_analysis:
            st.subheader("🤖 AI-Powered Performance Analysis")
            st.markdown(st.session_state.llm_analysis)
            
            # Option to regenerate analysis
            if st.button("🔄 Regenerate Analysis"):
                if api_key and st.session_state.stats and st.session_state.analysis_data:
                    with st.spinner("Getting fresh AI insights..."):
                        llm_analysis = get_llm_analysis(
                            st.session_state.stats,
                            st.session_state.analysis_data,
                            api_key
                        )
                        st.session_state.llm_analysis = llm_analysis
                        st.rerun()
        elif st.session_state.processed:
            st.warning("⚠️ Please enter your Google API key in the sidebar to get AI-powered insights.")
        else:
            st.info("👆 Upload and analyze a video first, then get AI-powered feedback here.")


if __name__ == "__main__":
    main()

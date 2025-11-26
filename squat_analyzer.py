"""
Squat analysis module to evaluate squat form and count repetitions.
"""
import numpy as np
from pose_detector import PoseDetector


class SquatAnalyzer:
    """Analyzes squat exercises from pose landmarks."""
    
    def __init__(self, depth_threshold=90, min_knee_angle=70):
        """
        Initialize the squat analyzer.
        
        Args:
            depth_threshold: Knee angle threshold for valid squat (degrees)
            min_knee_angle: Minimum knee angle to count as bottom position
        """
        self.depth_threshold = depth_threshold
        self.min_knee_angle = min_knee_angle
        self.squat_count = 0
        self.squat_state = "up"  # "up" or "down"
        
        # Store analysis data
        self.knee_angles = []
        self.hip_angles = []
        self.depths = []
        self.timestamps = []
        self.frame_count = 0
        
        # MediaPipe landmark indices
        self.LEFT_HIP = 23
        self.LEFT_KNEE = 25
        self.LEFT_ANKLE = 27
        self.LEFT_SHOULDER = 11
        self.RIGHT_HIP = 24
        self.RIGHT_KNEE = 26
        self.RIGHT_ANKLE = 28
        self.RIGHT_SHOULDER = 12
        
    def analyze_frame(self, landmarks, frame_width, frame_height, detector):
        """
        Analyze a single frame for squat form.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            frame_width: Width of the frame
            frame_height: Height of the frame
            detector: PoseDetector instance
            
        Returns:
            dict: Analysis results for this frame
        """
        if not landmarks:
            return None
        
        # Get landmark coordinates (using right side by default)
        hip = detector.get_landmark_coords(landmarks, self.RIGHT_HIP, frame_width, frame_height)
        knee = detector.get_landmark_coords(landmarks, self.RIGHT_KNEE, frame_width, frame_height)
        ankle = detector.get_landmark_coords(landmarks, self.RIGHT_ANKLE, frame_width, frame_height)
        shoulder = detector.get_landmark_coords(landmarks, self.RIGHT_SHOULDER, frame_width, frame_height)
        
        # Calculate angles
        knee_angle = detector.calculate_angle(hip, knee, ankle)
        hip_angle = detector.calculate_angle(shoulder, hip, knee)
        
        # Calculate depth (hip vertical position relative to knee)
        depth_percent = None
        if hip and knee:
            # Lower hip = deeper squat
            depth_percent = ((knee[1] - hip[1]) / frame_height) * 100
        
        # Store data
        self.frame_count += 1
        if knee_angle:
            self.knee_angles.append(knee_angle)
        if hip_angle:
            self.hip_angles.append(hip_angle)
        if depth_percent:
            self.depths.append(depth_percent)
        self.timestamps.append(self.frame_count)
        
        # Count squats
        if knee_angle:
            if knee_angle < self.depth_threshold and self.squat_state == "up":
                self.squat_state = "down"
            elif knee_angle > 160 and self.squat_state == "down":
                self.squat_state = "up"
                self.squat_count += 1
        
        return {
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "depth_percent": depth_percent,
            "squat_count": self.squat_count,
            "squat_state": self.squat_state,
            "knee_coord": knee,
            "hip_coord": hip,
            "ankle_coord": ankle
        }
    
    def get_form_feedback(self, knee_angle, depth_percent):
        """
        Provide form feedback based on analysis.
        
        Args:
            knee_angle: Current knee angle
            depth_percent: Current squat depth
            
        Returns:
            str: Feedback message
        """
        feedback = []
        
        if knee_angle:
            if knee_angle < self.min_knee_angle:
                feedback.append("Great depth!")
            elif knee_angle < self.depth_threshold:
                feedback.append("Good squat")
            elif knee_angle < 120:
                feedback.append("Go deeper")
            else:
                feedback.append("Standing")
        
        return " | ".join(feedback) if feedback else "Keep going"
    
    def get_summary_statistics(self):
        """
        Calculate summary statistics of the squat session.
        
        Returns:
            dict: Summary statistics
        """
        stats = {
            "total_squats": self.squat_count,
            "avg_knee_angle": np.mean(self.knee_angles) if self.knee_angles else 0,
            "min_knee_angle": np.min(self.knee_angles) if self.knee_angles else 0,
            "max_knee_angle": np.max(self.knee_angles) if self.knee_angles else 0,
            "avg_hip_angle": np.mean(self.hip_angles) if self.hip_angles else 0,
            "avg_depth": np.mean(self.depths) if self.depths else 0,
            "max_depth": np.max(self.depths) if self.depths else 0,
            "total_frames": self.frame_count
        }
        return stats
    
    def get_analysis_data(self):
        """
        Get all collected analysis data for visualization.
        
        Returns:
            dict: All analysis data
        """
        return {
            "knee_angles": self.knee_angles,
            "hip_angles": self.hip_angles,
            "depths": self.depths,
            "timestamps": self.timestamps,
            "squat_count": self.squat_count
        }

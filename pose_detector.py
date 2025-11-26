"""
Pose detection module using MediaPipe for video analysis.
"""
import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """Detects human pose landmarks in video frames."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect_pose(self, frame):
        """
        Detect pose landmarks in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            landmarks: Detected pose landmarks
            frame_rgb: Frame in RGB format
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        return results.pose_landmarks, frame_rgb
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw pose landmarks on the frame.
        
        Args:
            frame: Input frame
            landmarks: Pose landmarks to draw
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        return frame
    
    def get_landmark_coords(self, landmarks, landmark_id, frame_width, frame_height):
        """
        Get pixel coordinates of a specific landmark.
        
        Args:
            landmarks: Pose landmarks
            landmark_id: ID of the landmark to retrieve
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            tuple: (x, y) coordinates in pixels
        """
        if landmarks:
            landmark = landmarks.landmark[landmark_id]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            return (x, y)
        return None
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points.
        
        Args:
            point1, point2, point3: Points as (x, y) tuples
            
        Returns:
            float: Angle in degrees
        """
        if None in [point1, point2, point3]:
            return None
            
        # Convert to numpy arrays
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def close(self):
        """Release resources."""
        self.pose.close()

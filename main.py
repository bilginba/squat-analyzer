"""
Main application script for squat video analysis.
"""
import cv2
import argparse
import os
from pose_detector import PoseDetector
from squat_analyzer import SquatAnalyzer
from visualizer import SquatVisualizer


def process_video(video_path, output_video=True, show_preview=False):
    """
    Process a video file and analyze squat exercises.
    
    Args:
        video_path: Path to input video file
        output_video: Whether to save output video with annotations
        show_preview: Whether to show real-time preview
    """
    # Initialize components
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    analyzer = SquatAnalyzer(depth_threshold=90, min_knee_angle=70)
    visualizer = SquatVisualizer(output_dir="output")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"Processing Video: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Setup output video writer
    output_writer = None
    if output_video:
        output_path = os.path.join("output", "analyzed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output video will be saved to: {output_path}")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Detect pose
            landmarks, _ = detector.detect_pose(frame)
            
            # Analyze squat
            analysis = analyzer.analyze_frame(landmarks, frame_width, frame_height, detector)
            
            # Draw pose landmarks
            frame = detector.draw_landmarks(frame, landmarks)
            
            # Add analysis overlay
            if analysis:
                # Display squat count
                cv2.putText(frame, f"Squats: {analysis['squat_count']}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Display state
                state_color = (0, 255, 255) if analysis['squat_state'] == "down" else (255, 255, 255)
                cv2.putText(frame, f"State: {analysis['squat_state'].upper()}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
                
                # Display knee angle
                if analysis['knee_angle']:
                    cv2.putText(frame, f"Knee: {analysis['knee_angle']:.1f}deg", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    # Draw angle arc
                    if analysis['knee_coord']:
                        cv2.circle(frame, analysis['knee_coord'], 8, (0, 255, 255), -1)
                
                # Display hip angle
                if analysis['hip_angle']:
                    cv2.putText(frame, f"Hip: {analysis['hip_angle']:.1f}deg", 
                               (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Display form feedback
                feedback = analyzer.get_form_feedback(analysis['knee_angle'], analysis['depth_percent'])
                cv2.putText(frame, feedback, 
                           (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Progress indicator
            progress = (frame_count / total_frames) * 100
            cv2.rectangle(frame, (10, frame_height - 70), 
                         (int(10 + (frame_width - 20) * (progress / 100)), frame_height - 50), 
                         (0, 255, 0), -1)
            cv2.putText(frame, f"{progress:.1f}%", 
                       (frame_width - 150, frame_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            if output_writer:
                output_writer.write(frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('Squat Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nAnalysis interrupted by user.")
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processing: {progress:.1f}% - Squats detected: {analyzer.squat_count}", end='\r')
    
    finally:
        # Cleanup
        cap.release()
        if output_writer:
            output_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        detector.close()
    
    print(f"\n\n{'='*60}")
    print("Video Processing Complete!")
    print(f"{'='*60}\n")
    
    # Get analysis results
    stats = analyzer.get_summary_statistics()
    data = analyzer.get_analysis_data()
    
    # Print summary
    print("ANALYSIS SUMMARY:")
    print(f"  Total Squats: {stats['total_squats']}")
    print(f"  Average Knee Angle: {stats['avg_knee_angle']:.1f}°")
    print(f"  Minimum Knee Angle: {stats['min_knee_angle']:.1f}°")
    print(f"  Maximum Knee Angle: {stats['max_knee_angle']:.1f}°")
    print(f"  Average Hip Angle: {stats['avg_hip_angle']:.1f}°")
    print(f"  Maximum Depth: {stats['max_depth']:.1f}%")
    print(f"  Total Frames Processed: {stats['total_frames']}")
    
    # Generate visualizations
    visualizer.create_all_charts(data, stats, fps)
    
    print(f"\n{'='*60}")
    print("Analysis complete! Check the 'output' folder for results.")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI-powered squat video analysis')
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--no-output-video', action='store_true',
                       help='Skip saving output video with annotations')
    parser.add_argument('--preview', action='store_true',
                       help='Show real-time preview (press Q to quit)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Process the video
    process_video(
        video_path=args.video,
        output_video=not args.no_output_video,
        show_preview=args.preview
    )


if __name__ == "__main__":
    main()

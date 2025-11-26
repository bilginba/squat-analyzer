"""
Visualization module to create charts and graphs from squat analysis data.
"""
import matplotlib.pyplot as plt
import numpy as np
import os


class SquatVisualizer:
    """Creates visualizations for squat analysis data."""
    
    def __init__(self, output_dir="output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save output charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_angle_timeline(self, data, fps=30):
        """
        Plot knee and hip angles over time.
        
        Args:
            data: Analysis data dictionary
            fps: Frames per second of the video
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Convert frames to seconds
        time_seconds = np.array(data["timestamps"]) / fps
        
        # Plot knee angles
        if data["knee_angles"]:
            ax1.plot(time_seconds[:len(data["knee_angles"])], data["knee_angles"], 
                    'b-', linewidth=2, label='Knee Angle')
            ax1.axhline(y=90, color='r', linestyle='--', label='Depth Threshold (90°)')
            ax1.set_ylabel('Angle (degrees)', fontsize=12)
            ax1.set_title('Knee Angle Over Time', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot hip angles
        if data["hip_angles"]:
            ax2.plot(time_seconds[:len(data["hip_angles"])], data["hip_angles"], 
                    'g-', linewidth=2, label='Hip Angle')
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Angle (degrees)', fontsize=12)
            ax2.set_title('Hip Angle Over Time', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'angle_timeline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved angle timeline chart to: {output_path}")
        
    def plot_depth_analysis(self, data, fps=30):
        """
        Plot squat depth analysis.
        
        Args:
            data: Analysis data dictionary
            fps: Frames per second of the video
        """
        if not data["depths"]:
            print("No depth data available for visualization")
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert frames to seconds
        time_seconds = np.array(data["timestamps"][:len(data["depths"])]) / fps
        
        # Plot depth
        ax.plot(time_seconds, data["depths"], 'purple', linewidth=2, label='Squat Depth')
        ax.fill_between(time_seconds, 0, data["depths"], alpha=0.3, color='purple')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Depth (%)', fontsize=12)
        ax.set_title('Squat Depth Analysis Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'depth_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved depth analysis chart to: {output_path}")
        
    def plot_summary_dashboard(self, stats, data):
        """
        Create a summary dashboard with multiple metrics.
        
        Args:
            stats: Summary statistics dictionary
            data: Analysis data dictionary
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Squat count
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, f"{stats['total_squats']}", 
                ha='center', va='center', fontsize=60, fontweight='bold', color='#2E86AB')
        ax1.text(0.5, 0.15, 'Total Squats', 
                ha='center', va='center', fontsize=16, color='gray')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Average knee angle
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, f"{stats['avg_knee_angle']:.1f}°", 
                ha='center', va='center', fontsize=48, fontweight='bold', color='#A23B72')
        ax2.text(0.5, 0.15, 'Avg Knee Angle', 
                ha='center', va='center', fontsize=16, color='gray')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Knee angle distribution
        ax3 = fig.add_subplot(gs[1, :])
        if data["knee_angles"]:
            ax3.hist(data["knee_angles"], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax3.axvline(x=stats['avg_knee_angle'], color='red', linestyle='--', 
                       linewidth=2, label=f'Average: {stats["avg_knee_angle"]:.1f}°')
            ax3.set_xlabel('Knee Angle (degrees)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Knee Angle Distribution', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Depth box plot
        ax4 = fig.add_subplot(gs[2, 0])
        if data["depths"]:
            ax4.boxplot(data["depths"], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#F18F01', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax4.set_ylabel('Depth (%)', fontsize=12)
            ax4.set_title('Squat Depth Distribution', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Statistics table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Squats', f"{stats['total_squats']}"],
            ['Avg Knee Angle', f"{stats['avg_knee_angle']:.1f}°"],
            ['Min Knee Angle', f"{stats['min_knee_angle']:.1f}°"],
            ['Max Knee Angle', f"{stats['max_knee_angle']:.1f}°"],
            ['Avg Hip Angle', f"{stats['avg_hip_angle']:.1f}°"],
            ['Max Depth', f"{stats['max_depth']:.1f}%"],
        ]
        
        table = ax5.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(stats_data)):
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle('Squat Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        output_path = os.path.join(self.output_dir, 'summary_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary dashboard to: {output_path}")
        
    def create_all_charts(self, data, stats, fps=30):
        """
        Generate all visualization charts.
        
        Args:
            data: Analysis data dictionary
            stats: Summary statistics dictionary
            fps: Frames per second of the video
        """
        print("\nGenerating visualization charts...")
        self.plot_angle_timeline(data, fps)
        self.plot_depth_analysis(data, fps)
        self.plot_summary_dashboard(stats, data)
        print("All charts generated successfully!")

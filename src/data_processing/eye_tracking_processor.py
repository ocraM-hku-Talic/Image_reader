import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path

class EyeTrackingProcessor:
    """
    Process eye tracking data from Quest 3 VR headset.
    Converts raw eye tracking data into heatmaps for analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize the eye tracking processor.
        
        Args:
            config (dict, optional): Configuration parameters.
        """
        self.config = config or {}
        self.resolution = self.config.get("resolution", (1920, 1080))
        self.smoothing_factor = self.config.get("smoothing_factor", 25)
        self.color_map = self.config.get("color_map", "jet")
    
    def load_eye_tracking_data(self, file_path):
        """
        Load eye tracking data from Quest 3 VR session.
        
        Args:
            file_path (str): Path to the eye tracking data file.
            
        Returns:
            pd.DataFrame: DataFrame containing eye gaze positions and timestamps.
        """
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Convert JSON data to DataFrame based on Quest 3 format
            # This will need to be adjusted based on the actual Quest 3 data format
            df = pd.DataFrame(data["eyeTrackingData"])
            return df
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def generate_heatmap(self, gaze_points, output_path=None, background_image=None):
        """
        Generate heatmap from eye tracking gaze points.
        
        Args:
            gaze_points (pd.DataFrame): DataFrame with columns 'x' and 'y' for gaze positions.
            output_path (str, optional): Path to save the generated heatmap.
            background_image (str, optional): Path to background image to overlay heatmap on.
            
        Returns:
            np.ndarray: Generated heatmap as an image.
        """
        # Create an empty heatmap
        heatmap = np.zeros(self.resolution, dtype=np.float32)
        
        # Add gaussian blur for each gaze point
        for _, point in gaze_points.iterrows():
            x, y = int(point['x'] * self.resolution[0]), int(point['y'] * self.resolution[1])
            if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
                # Create a small matrix with a single point
                tmp_heatmap = np.zeros(self.resolution, dtype=np.float32)
                tmp_heatmap[y, x] = 1
                # Apply gaussian blur
                tmp_heatmap = cv2.GaussianBlur(tmp_heatmap, (self.smoothing_factor, self.smoothing_factor), 0)
                # Add to the main heatmap
                heatmap += tmp_heatmap
        
        # Normalize heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, getattr(cv2, f'COLORMAP_{self.color_map.upper()}'))
        
        # If background image is provided, overlay heatmap on it
        if background_image is not None:
            if isinstance(background_image, str):
                bg = cv2.imread(background_image)
                bg = cv2.resize(bg, self.resolution)
            else:
                bg = background_image
                
            # Overlay heatmap on background
            result = cv2.addWeighted(bg, 0.7, heatmap_colored, 0.3, 0)
        else:
            result = heatmap_colored
        
        # Save heatmap if output_path is provided
        if output_path:
            cv2.imwrite(output_path, result)
        
        return result
    
    def analyze_gaze_patterns(self, gaze_points):
        """
        Analyze gaze patterns for metrics on public speaking performance.
        
        Args:
            gaze_points (pd.DataFrame): DataFrame with eye tracking data.
            
        Returns:
            dict: Dictionary with analysis metrics.
        """
        # Calculate audience coverage
        x_range = np.ptp(gaze_points['x'])
        y_range = np.ptp(gaze_points['y'])
        coverage = (x_range * y_range) / (1.0 * 1.0)  # Normalize by total possible area
        
        # Calculate fixation duration statistics
        # Assuming consecutive points within threshold represent a fixation
        fixation_threshold = 0.05  # Spatial threshold for fixation
        fixation_durations = []
        current_fixation = []
        
        for i in range(1, len(gaze_points)):
            prev_point = gaze_points.iloc[i-1]
            curr_point = gaze_points.iloc[i]
            
            dist = np.sqrt((curr_point['x'] - prev_point['x'])**2 + (curr_point['y'] - prev_point['y'])**2)
            
            if dist < fixation_threshold:
                if not current_fixation:
                    current_fixation = [prev_point]
                current_fixation.append(curr_point)
            else:
                if current_fixation:
                    duration = current_fixation[-1]['timestamp'] - current_fixation[0]['timestamp']
                    fixation_durations.append(duration)
                    current_fixation = []
        
        # Calculate metrics
        metrics = {
            'audience_coverage': coverage,
            'avg_fixation_duration': np.mean(fixation_durations) if fixation_durations else 0,
            'max_fixation_duration': np.max(fixation_durations) if fixation_durations else 0,
            'num_fixations': len(fixation_durations),
        }
        
        return metrics
    
    def process_session(self, file_path, output_dir=None, background_image=None):
        """
        Process an entire eye tracking session.
        
        Args:
            file_path (str): Path to eye tracking data file.
            output_dir (str, optional): Directory to save processing results.
            background_image (str, optional): Path to background image for heatmap overlay.
            
        Returns:
            dict: Dictionary with heatmap and analysis metrics.
        """
        # Create output directory if needed
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        # Load eye tracking data
        gaze_data = self.load_eye_tracking_data(file_path)
        
        # Generate heatmap
        heatmap_path = None
        if output_dir:
            heatmap_path = str(Path(output_dir) / "eye_tracking_heatmap.png")
            
        heatmap = self.generate_heatmap(gaze_data, heatmap_path, background_image)
        
        # Analyze gaze patterns
        metrics = self.analyze_gaze_patterns(gaze_data)
        
        # Save metrics if output directory is provided
        if output_dir:
            metrics_path = str(Path(output_dir) / "eye_tracking_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return {
            'heatmap': heatmap,
            'metrics': metrics,
            'heatmap_path': heatmap_path if output_dir else None
        }
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import seaborn as sns


class HeatmapVisualizer:
    """
    Visualize eye tracking heatmaps and metrics for public speaking analysis.
    Provides tools for rendering heatmaps with different visualization options.
    """
    
    def __init__(self, config=None):
        """
        Initialize the heatmap visualizer.
        
        Args:
            config (dict, optional): Configuration parameters.
        """
        self.config = config or {}
        self.figsize = self.config.get("figsize", (12, 8))
        self.dpi = self.config.get("dpi", 100)
        self.cmap = self.config.get("cmap", "jet")
    
    def plot_heatmap(self, heatmap, title=None, output_path=None):
        """
        Plot a heatmap image.
        
        Args:
            heatmap (np.ndarray): Heatmap image array.
            title (str, optional): Title for the plot.
            output_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        else:
            heatmap_rgb = heatmap
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Display the heatmap
        ax.imshow(heatmap_rgb)
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=14)
        
        # Remove axes for cleaner look
        ax.axis('off')
        
        # Save figure if output path is provided
        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def visualize_gaze_metrics(self, metrics, output_path=None):
        """
        Visualize gaze metrics with bar charts and other visualizations.
        
        Args:
            metrics (dict): Dictionary containing eye tracking metrics.
            output_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        # Set style
        sns.set_style("whitegrid")
        
        # Plot audience coverage
        axes[0].bar(['Audience Coverage'], [metrics.get('audience_coverage', 0)], color='skyblue')
        axes[0].set_ylim(0, 1.0)
        axes[0].set_title('Audience Coverage')
        axes[0].set_ylabel('Coverage Ratio')
        
        # Plot average fixation duration
        axes[1].bar(['Avg. Fixation Duration'], [metrics.get('avg_fixation_duration', 0)], color='lightgreen')
        axes[1].set_title('Average Fixation Duration')
        axes[1].set_ylabel('Duration (seconds)')
        
        # Plot maximum fixation duration
        axes[2].bar(['Max Fixation Duration'], [metrics.get('max_fixation_duration', 0)], color='salmon')
        axes[2].set_title('Maximum Fixation Duration')
        axes[2].set_ylabel('Duration (seconds)')
        
        # Plot number of fixations
        axes[3].bar(['Number of Fixations'], [metrics.get('num_fixations', 0)], color='mediumpurple')
        axes[3].set_title('Number of Fixations')
        axes[3].set_ylabel('Count')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def create_dashboard(self, heatmap, metrics, output_path=None):
        """
        Create a comprehensive dashboard with heatmap and metrics.
        
        Args:
            heatmap (np.ndarray): Heatmap image array.
            metrics (dict): Dictionary containing eye tracking metrics.
            output_path (str, optional): Path to save the dashboard.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        else:
            heatmap_rgb = heatmap
        
        # Create figure with gridspec for custom layout
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        gs = fig.add_gridspec(2, 3)
        
        # Main heatmap (spans top row)
        ax_heatmap = fig.add_subplot(gs[0, :])
        ax_heatmap.imshow(heatmap_rgb)
        ax_heatmap.set_title('Eye Tracking Heatmap', fontsize=16)
        ax_heatmap.axis('off')
        
        # Metrics plots (bottom row)
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.bar(['Audience Coverage'], [metrics.get('audience_coverage', 0)], color='skyblue')
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Audience Coverage')
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.bar(['Avg. Fixation'], [metrics.get('avg_fixation_duration', 0)], color='lightgreen')
        ax2.bar(['Max. Fixation'], [metrics.get('max_fixation_duration', 0)], color='salmon')
        ax2.set_title('Fixation Duration (seconds)')
        
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.bar(['Number of Fixations'], [metrics.get('num_fixations', 0)], color='mediumpurple')
        ax3.set_title('Fixation Count')
        
        # Add feedback text based on metrics
        feedback = self._generate_basic_feedback(metrics)
        fig.text(0.5, 0.02, feedback, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save dashboard if output path is provided
        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def _generate_basic_feedback(self, metrics):
        """
        Generate basic feedback based on metrics.
        This will be enhanced by the Ollama model later.
        
        Args:
            metrics (dict): Dictionary containing eye tracking metrics.
            
        Returns:
            str: Basic feedback message.
        """
        audience_coverage = metrics.get('audience_coverage', 0)
        avg_fixation = metrics.get('avg_fixation_duration', 0)
        num_fixations = metrics.get('num_fixations', 0)
        
        feedback_parts = []
        
        if audience_coverage < 0.3:
            feedback_parts.append("You need to improve audience engagement by looking at more areas.")
        elif audience_coverage > 0.7:
            feedback_parts.append("Great job scanning the entire audience!")
        
        if avg_fixation > 3.0:
            feedback_parts.append("Try to reduce time spent fixating on specific audience members.")
        elif avg_fixation < 0.5 and num_fixations > 50:
            feedback_parts.append("Your gaze appears too rapid. Try to maintain more steady eye contact.")
        
        if not feedback_parts:
            feedback_parts.append("Your eye movement patterns look good for public speaking.")
            
        return " ".join(feedback_parts)
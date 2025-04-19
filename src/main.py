import argparse
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.eye_tracking_processor import EyeTrackingProcessor
from src.model.ollama_model import OllamaEyeTrackingAnalyzer
from src.visualization.heatmap_visualizer import HeatmapVisualizer


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Eye Tracking Analysis for VR Public Speaking Training"
    )
    
    # Input options
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--image", type=str, help="Path to heatmap image file"
    )
    input_group.add_argument(
        "--data", type=str, help="Path to eye tracking data file (JSON or CSV)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output", type=str, default="./output", help="Directory to save output files"
    )
    output_group.add_argument(
        "--no-display", action="store_true", help="Don't display visualizations"
    )
    
    # Model options
    model_group = parser.add_argument_group("Model options")
    model_group.add_argument(
        "--model", type=str, default="llava:7b", help="Ollama model to use"
    )
    model_group.add_argument(
        "--api-base", type=str, default="http://localhost:11434", help="Ollama API base URL"
    )
    
    # Fine-tuning options
    finetune_group = parser.add_argument_group("Fine-tuning options")
    finetune_group.add_argument(
        "--finetune", action="store_true", help="Fine-tune the model"
    )
    finetune_group.add_argument(
        "--training-data", type=str, help="Directory with training data"
    )
    finetune_group.add_argument(
        "--annotations", type=str, help="JSON file with expert annotations"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    processor = EyeTrackingProcessor()
    visualizer = HeatmapVisualizer()
    analyzer = OllamaEyeTrackingAnalyzer(model_name=args.model, api_base=args.api_base)
    
    # Handle fine-tuning mode
    if args.finetune:
        if not args.training_data or not args.annotations:
            print("Error: --training-data and --annotations required for fine-tuning")
            return
            
        # Prepare fine-tuning data
        output_training_dir = output_dir / "training_data"
        print(f"Preparing training data in {output_training_dir}...")
        sample_count = analyzer.prepare_training_data(
            args.training_data, args.annotations, str(output_training_dir)
        )
        print(f"Created {sample_count} training samples")
        
        # Start fine-tuning
        print(f"Fine-tuning model based on {args.model}...")
        result = analyzer.fine_tune_model(
            str(output_training_dir), 
            model_destination="eye-tracking-analyzer",
            base_model=args.model
        )
        
        if result.get("status") == "success":
            print(f"Success: {result.get('message')}")
        else:
            print(f"Error: {result.get('message')}")
        
        return
    
    # Handle heatmap analysis mode
    heatmap = None
    
    if args.image:
        # Load heatmap directly from image
        print(f"Loading heatmap from {args.image}")
        heatmap = cv2.imread(args.image)
        if heatmap is None:
            print(f"Error: Could not load image {args.image}")
            return
    elif args.data:
        # Generate heatmap from eye tracking data
        print(f"Processing eye tracking data from {args.data}")
        try:
            # Load eye tracking data
            gaze_data = processor.load_eye_tracking_data(args.data)
            
            # Generate heatmap
            heatmap_path = str(output_dir / "generated_heatmap.png") if args.output else None
            heatmap = processor.generate_heatmap(gaze_data, output_path=heatmap_path)
            
            # Analyze gaze patterns
            metrics = processor.analyze_gaze_patterns(gaze_data)
            
            # Create and save visualization
            if not args.no_display or args.output:
                dashboard_path = str(output_dir / "dashboard.png") if args.output else None
                fig = visualizer.create_dashboard(heatmap, metrics, output_path=dashboard_path)
                if not args.no_display:
                    plt.show()
                
        except Exception as e:
            print(f"Error processing eye tracking data: {e}")
            return
    else:
        print("Error: Either --image or --data must be provided")
        return
    
    # Analyze heatmap using Ollama model
    if heatmap is not None:
        print(f"Analyzing heatmap with Ollama model '{args.model}'...")
        try:
            analysis = analyzer.analyze_heatmap(image_array=heatmap)
            
            if "error" in analysis:
                print(f"Error from Ollama API: {analysis['error']}")
            else:
                print("\n===== FEEDBACK =====")
                print(analysis["feedback"])
                
                # Save feedback to file if output is specified
                if args.output:
                    feedback_path = output_dir / "feedback.txt"
                    with open(feedback_path, "w") as f:
                        f.write(analysis["feedback"])
                    print(f"Feedback saved to {feedback_path}")
                    
                # Print model information
                print("\n===== MODEL INFO =====")
                for key, value in analysis["model_info"].items():
                    print(f"{key}: {value}")
                    
        except Exception as e:
            print(f"Error analyzing heatmap: {e}")


if __name__ == "__main__":
    main()
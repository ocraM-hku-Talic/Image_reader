import argparse
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.eye_tracking_processor import EyeTrackingProcessor
from src.model.ollama_model import OllamaEyeTrackingAnalyzer
from src.visualization.heatmap_visualizer import HeatmapVisualizer


def process_heatmap_folder(heatmap_folder_path, output_dir=None, model_name="llava:7b", api_base="http://localhost:11434"):
    """
    Process all heatmap images in a folder and generate feedback using the Ollama model.
    
    Args:
        heatmap_folder_path (str): Path to the folder containing heatmap images
        output_dir (str, optional): Directory to save the results
        model_name (str): Name of the Ollama model to use
        api_base (str): Base URL for the Ollama API
        
    Returns:
        dict: Dictionary with results for each processed image
    """
    # Initialize components
    analyzer = OllamaEyeTrackingAnalyzer(model_name=model_name, api_base=api_base)
    visualizer = HeatmapVisualizer()
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files in the folder
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(heatmap_folder_path, ext)))
    
    if not image_files:
        print(f"No image files found in {heatmap_folder_path}")
        return {}
    
    results = {}
    
    # Process each image
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        print(f"Processing {img_filename}...")
        
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            continue
        
        # Analyze with Ollama model
        try:
            analysis = analyzer.analyze_heatmap(image_array=img)
            
            # Save the results
            results[img_filename] = analysis
            
            if output_dir:
                # Save feedback to text file
                feedback_path = output_path / f"{os.path.splitext(img_filename)[0]}_feedback.txt"
                with open(feedback_path, "w") as f:
                    f.write(analysis.get("feedback", "No feedback available"))
                
                # Create visualization with feedback
                if "error" not in analysis:
                    output_img_path = output_path / f"{os.path.splitext(img_filename)[0]}_analyzed.png"
                    # Draw the feedback on the image
                    img_with_text = img.copy()
                    feedback_text = analysis.get("feedback", "No feedback available")
                    
                    # Split feedback into lines for better display
                    max_line_length = 80
                    lines = []
                    words = feedback_text.split()
                    current_line = words[0]
                    
                    for word in words[1:]:
                        if len(current_line + " " + word) <= max_line_length:
                            current_line += " " + word
                        else:
                            lines.append(current_line)
                            current_line = word
                    
                    lines.append(current_line)
                    
                    # Create a taller image to accommodate text
                    padding = 50 + 30 * len(lines)  # Base padding + space per line
                    img_with_text = cv2.copyMakeBorder(img_with_text, 0, padding, 0, 0, 
                                                     cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    
                    # Add the feedback text
                    for i, line in enumerate(lines):
                        cv2.putText(img_with_text, line, 
                                    (10, img.shape[0] + 30 + 30*i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    cv2.imwrite(str(output_img_path), img_with_text)
                    print(f"Analysis saved to {output_img_path}")
            
        except Exception as e:
            print(f"Error analyzing image {img_path}: {e}")
            results[img_filename] = {"error": str(e)}
    
    return results


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
    input_group.add_argument(
        "--folder", type=str, help="Path to folder containing heatmap images"
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
    
    # Process folder of heatmap images
    if args.folder:
        print(f"Processing heatmap images in folder: {args.folder}")
        results = process_heatmap_folder(
            args.folder, 
            output_dir=args.output, 
            model_name=args.model, 
            api_base=args.api_base
        )
        print(f"Processed {len(results)} images")
        return
    
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
        print("Error: Either --image, --data, or --folder must be provided")
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
# VR Public Speaking Trainer - Eye Tracking Analysis

## Project Overview
This project uses Quest 3 VR for public speaking training and provides analysis of user eye tracking data. After a training session, a heatmap of the user's eye tracking patterns is generated, and a fine-tuned Ollama model provides feedback on the user's performance.

## Features
- Integration with Quest 3 VR for public speaking training scenarios
- Eye tracking data collection and heatmap generation
- Analysis of eye tracking patterns using a fine-tuned Ollama model
- Personalized feedback on public speaking performance based on gaze patterns
- Visualization tools for eye tracking metrics and heatmaps
- Model fine-tuning workflow for custom feedback generation

## Technical Architecture

### Project Structure
```
Image_reader/
├── README.md
├── requirements.txt
├── data/              # For storing eye tracking data and sample images
├── src/
│   ├── main.py        # Main application entry point
│   ├── data_processing/
│   │   └── eye_tracking_processor.py  # Processes eye tracking data into heatmaps
│   ├── visualization/
│   │   └── heatmap_visualizer.py     # Visualization tools for heatmaps and metrics
│   └── model/
│       ├── __init__.py
│       └── ollama_model.py           # Ollama model integration for analysis
```

### Components
- **VR Component**: Quest 3 VR headset with eye tracking capabilities
- **Data Processing**: Python scripts for processing eye tracking data and generating heatmaps
- **Model**: Fine-tuned Ollama model for image analysis and feedback generation
- **Visualization**: Tools for rendering heatmaps and metrics dashboards

## Setup and Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running locally

### Installation Steps
1. Clone this repository
2. Set up a Python virtual environment (recommended)
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install Ollama from [ollama.ai](https://ollama.ai)
5. Pull the required model:
   ```
   ollama pull llava:7b
   ```

## Usage

### Analyzing Eye Tracking Data
You can analyze eye tracking data either from raw gaze data or from pre-generated heatmap images:

#### From Raw Eye Tracking Data
```
python src/main.py --data path/to/eye_tracking_data.json --output ./results
```

#### From Heatmap Images
```
python src/main.py --image path/to/heatmap.png --output ./results
```

### Fine-Tuning the Model
To fine-tune the Ollama model with custom training data:

1. Prepare a directory with heatmap images
2. Create a JSON file with expert annotations for each image
3. Run the fine-tuning process:
   ```
   python src/main.py --finetune --training-data path/to/heatmaps --annotations path/to/annotations.json --output ./fine_tune_results
   ```

### Command Line Options
```
usage: main.py [-h] [--image IMAGE] [--data DATA] [--output OUTPUT]
               [--no-display] [--model MODEL] [--api-base API_BASE]
               [--finetune] [--training-data TRAINING_DATA]
               [--annotations ANNOTATIONS]

Eye Tracking Analysis for VR Public Speaking Training

Input options:
  --image IMAGE          Path to heatmap image file
  --data DATA            Path to eye tracking data file (JSON or CSV)

Output options:
  --output OUTPUT        Directory to save output files
  --no-display           Don't display visualizations

Model options:
  --model MODEL          Ollama model to use
  --api-base API_BASE    Ollama API base URL

Fine-tuning options:
  --finetune             Fine-tune the model
  --training-data TRAINING_DATA
                         Directory with training data
  --annotations ANNOTATIONS
                         JSON file with expert annotations
```

## Data Formats

### Eye Tracking Data Format
The system expects eye tracking data in one of the following formats:

#### JSON Format
```json
{
  "eyeTrackingData": [
    {"x": 0.5, "y": 0.3, "timestamp": 0.0},
    {"x": 0.51, "y": 0.32, "timestamp": 0.033},
    ...
  ]
}
```

#### CSV Format
```
x,y,timestamp
0.5,0.3,0.0
0.51,0.32,0.033
...
```

### Annotations Format for Fine-tuning
```json
{
  "heatmap1.png": {
    "expert_feedback": "The speaker shows good coverage of the audience on the left side, but needs to engage more with the right side. Consider practicing a more balanced gaze pattern."
  },
  "heatmap2.png": {
    "expert_feedback": "Excellent audience engagement with balanced coverage across the room. The speaker maintains appropriate fixation duration, creating good connection without staring."
  }
}
```

## Contributing
Contributions to improve the system are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
[MIT License](LICENSE)
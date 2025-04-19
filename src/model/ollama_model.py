import requests
import json
import base64
import os
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from pathlib import Path

class OllamaEyeTrackingAnalyzer:
    """
    Class to interact with a fine-tuned Ollama model for eye tracking analysis.
    Processes heatmap images and returns public speaking feedback.
    """
    
    def __init__(self, model_name="llava:7b", api_base="http://localhost:11434"):
        """
        Initialize the Ollama model interface.
        
        Args:
            model_name (str): Name of the Ollama model to use.
            api_base (str): Base URL for the Ollama API.
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_url = f"{api_base}/api/generate"
        
    def encode_image(self, image_path=None, image_array=None):
        """
        Encode image to base64 for API transmission.
        
        Args:
            image_path (str, optional): Path to the image file.
            image_array (np.ndarray, optional): Image as numpy array.
            
        Returns:
            str: Base64 encoded image.
        """
        if image_path:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif image_array is not None:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            img = Image.fromarray(image_array)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("Either image_path or image_array must be provided")
    
    def analyze_heatmap(self, image_path=None, image_array=None):
        """
        Analyze the heatmap using the Ollama model.
        
        Args:
            image_path (str, optional): Path to the heatmap image.
            image_array (np.ndarray, optional): Heatmap as numpy array.
            
        Returns:
            dict: Model analysis results.
        """
        # Encode the image
        encoded_image = self.encode_image(image_path, image_array)
        
        # Prepare the prompt for analysis
        system_prompt = """You are an expert in analyzing eye tracking heatmaps for public speaking. 
        Analyze this heatmap and provide feedback on the speaker's eye contact patterns. 
        Consider audience coverage, fixation points, and balanced engagement. 
        Provide specific suggestions for improvement."""
        
        # Prepare the API request
        payload = {
            "model": self.model_name,
            "prompt": "Analyze this eye tracking heatmap from a public speaking session and provide professional feedback:",
            "system": system_prompt,
            "images": [encoded_image],
            "stream": False
        }
        
        # Send the request to Ollama API
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            result = response.json()
            return {
                "feedback": result.get("response", "").strip(),
                "model_info": {
                    "model": self.model_name,
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                }
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "feedback": "Error connecting to Ollama API. Please check if the service is running."
            }
    
    def fine_tune_model(self, training_data_dir, model_destination="eye-tracking-analyzer", base_model="llava:7b"):
        """
        Fine-tune an Ollama model on eye tracking data.
        
        Args:
            training_data_dir (str): Directory containing training data.
            model_destination (str): Name for the fine-tuned model.
            base_model (str): Base model to start from.
            
        Returns:
            dict: Status of the fine-tuning process.
        """
        # Create a Modelfile for Ollama
        modelfile_path = os.path.join(training_data_dir, "Modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(f"""FROM {base_model}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "\\n\\n"
SYSTEM """You are an expert at analyzing eye tracking heatmaps from public speaking sessions. Your feedback is specific, actionable, and tailored for helping speakers improve their audience engagement through better eye contact patterns.\"
""")
            
        # Create the model
        try:
            create_url = f"{self.api_base}/api/create"
            with open(modelfile_path, 'r') as f:
                modelfile_content = f.read()
                
            payload = {
                "name": model_destination,
                "modelfile": modelfile_content,
                "path": training_data_dir
            }
            
            response = requests.post(create_url, json=payload)
            response.raise_for_status()
            
            return {
                "status": "success",
                "model_name": model_destination,
                "message": f"Model {model_destination} created successfully."
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Error creating model: {str(e)}"
            }
    
    def prepare_training_data(self, heatmaps_dir, annotations_file, output_dir):
        """
        Prepare training data for fine-tuning the Ollama model.
        
        Args:
            heatmaps_dir (str): Directory containing heatmap images.
            annotations_file (str): JSON file with expert annotations.
            output_dir (str): Directory to save prepared training data.
            
        Returns:
            int: Number of training samples prepared.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Create training samples
        sample_count = 0
        for filename, annotation in annotations.items():
            image_path = os.path.join(heatmaps_dir, filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping")
                continue
                
            # Create a training prompt with feedback
            training_sample = {
                "image": image_path,
                "prompt": "Analyze this eye tracking heatmap:",
                "response": annotation["expert_feedback"]
            }
            
            # Save the training sample
            with open(os.path.join(output_dir, f"sample_{sample_count}.json"), 'w') as f:
                json.dump(training_sample, f, indent=2)
                
            sample_count += 1
            
        return sample_count
    
    def preprocess_image(self, image_path=None, image_array=None, target_size=(224, 224)):
        """
        Preprocess image for model input.
        
        Args:
            image_path (str, optional): Path to the image file.
            image_array (np.ndarray, optional): Image as numpy array.
            target_size (tuple): Target size for the model input.
            
        Returns:
            np.ndarray: Preprocessed image.
        """
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image from {image_path}")
        elif image_array is not None:
            img = image_array
        else:
            raise ValueError("Either image_path or image_array must be provided")
        
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Convert to RGB if needed
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_resized
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        return img_normalized
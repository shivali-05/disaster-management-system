import os
import yaml
from ultralytics import YOLO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detection.model import get_detection_model

def create_dataset_yaml(base_dir):
    """
    Creates a data.yaml file required by YOLO for training on xView subset.
    """
    yaml_path = os.path.join(base_dir, "xview_subset.yaml")
    
    # Assuming prepare_data.py has split the xView dataset into train/val
    data = {
        'path': base_dir,
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'Building',
            1: 'Small Vehicle',
            2: 'Bridge',
            3: 'Tower'
            # (Truncated for the subset)
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
    return yaml_path

def train():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "xview"))
    
    if not os.path.exists(os.path.join(base_dir, "train", "images")):
        print(f"Processed training data not found at {base_dir}.")
        print("Please ensure xView data is downloaded and preprocessed.")
        return
        
    yaml_path = create_dataset_yaml(base_dir)
    print(f"Created YOLO configuration at {yaml_path}")
    
    print("Loading YOLOv8 model...")
    model = get_detection_model("yolov8n.pt") # Using nano model for the baseline prototype
    
    print("Starting YOLO training...")
    # These parameters can be tuned. Limiting epochs for prototyping.
    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        batch=4,
        project=os.path.join(os.path.dirname(__file__), "..", "models", "detection"),
        name="xview_baseline"
    )
    print("Training completed. Checkpoints are saved in models/detection/xview_baseline/weights/")

if __name__ == "__main__":
    train()

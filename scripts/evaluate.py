import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.dataset import SegmentationDataset
from src.segmentation.model import get_segmentation_model

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "loveda")
    val_img_dir = os.path.join(base_dir, "val", "images")
    val_mask_dir = os.path.join(base_dir, "val", "masks")
    
    if not os.path.exists(val_img_dir):
        print(f"Validation data not found at {val_img_dir}.")
        return
        
    dataset = SegmentationDataset(val_img_dir, val_mask_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    model = get_segmentation_model(num_classes=7).to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "models", "segmentation", "model_final.pt")
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No trained checkpoint found. Evaluating with initialized weights.")
        
    model.eval()
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            
            correct += (preds == masks).sum().item()
            total += masks.numel()
            
    pixel_acc = correct / total
    print(f"Validation Pixel Accuracy: {pixel_acc:.4f}")

if __name__ == "__main__":
    evaluate()

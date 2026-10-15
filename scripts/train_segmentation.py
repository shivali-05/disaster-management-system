import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.dataset import SegmentationDataset
from src.segmentation.model import get_segmentation_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "loveda")
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_mask_dir = os.path.join(base_dir, "train", "masks")
    
    if not os.path.exists(train_img_dir):
        print(f"Training data not found at {train_img_dir}.")
        print("Please run scripts/download_data.py and scripts/prepare_data.py first.")
        return
        
    dataset = SegmentationDataset(train_img_dir, train_mask_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    model = get_segmentation_model(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 2
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")
        
    # Save checkpoint
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models", "segmentation"), exist_ok=True)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "models", "segmentation", "model_final.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train()

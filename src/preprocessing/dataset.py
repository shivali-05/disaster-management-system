import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Basic transforms
        # Convert to tensor and normalize image
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask is class indices
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return image, mask

if __name__ == "__main__":
    # Test the dataset
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "loveda", "train")
    img_dir = os.path.join(processed_dir, "images")
    mask_dir = os.path.join(processed_dir, "masks")
    if os.path.exists(img_dir):
        ds = SegmentationDataset(img_dir, mask_dir)
        print(f"Dataset length: {len(ds)}")
        img, mask = ds[0]
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")

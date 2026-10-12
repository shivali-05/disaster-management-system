import torch
import torch.nn as nn

# Compatibility patch for PyTorch 2.1.2 vs newer Transformers pytree API
import torch.utils._pytree as _pytree
if not hasattr(_pytree, "register_pytree_node"):
    def _compat_register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=None, **kwargs):
        if hasattr(_pytree, "_register_pytree_node"):
            return _pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
    _pytree.register_pytree_node = _compat_register_pytree_node

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_earthvl_model import EarthVLMultimodalModel

_SEGMENTATION_MODEL = None

def get_loaded_segmentation_model():
    """Singleton getter for PyTorch segmentation model."""
    global _SEGMENTATION_MODEL
    if _SEGMENTATION_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EarthVLMultimodalModel(num_classes=9)
        
        # Load the custom trained model if it exists
        checkpoint_path = ROOT / "models" / "earthvl_custom.pt"
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded trained model from {checkpoint_path}")
        else:
            print("Warning: Training checkpoint not found. Using untrained weights.")
            
        model.to(device)
        model.eval()
        _SEGMENTATION_MODEL = model
    return _SEGMENTATION_MODEL

def predict_segmentation_mask(img_rgb):
    """
    Executes PyTorch semantic segmentation inference on an RGB numpy image.
    Returns: (mask: np.ndarray [H, W], confidence: float)
    """
    import numpy as np
    import cv2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_loaded_segmentation_model()
    
    h, w = img_rgb.shape[:2]
    # Resize to standard model resolution 512x512 for fast tensor inference
    resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    tensor_in = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor_in = (tensor_in - mean) / std
    tensor_in = tensor_in.to(device)
    
    with torch.no_grad():
        output = model(tensor_in)
        # If the model returns a dict (like torchvision FCN), extract 'out'
        if isinstance(output, dict):
            output = output["out"]
        probs = torch.softmax(output, dim=1)
        max_probs, argmax_mask = torch.max(probs, dim=1)
        
        mask_512 = argmax_mask.squeeze(0).cpu().numpy().astype(np.uint8)
        conf = float(max_probs.mean().cpu().item())
        
    # Resize mask back to original input dimensions
    mask_orig = cv2.resize(mask_512, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_orig, round(conf, 3)

if __name__ == "__main__":
    import numpy as np
    model = get_segmentation_model()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out['out'].shape}") # Expected: [2, 7, 512, 512]
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask, conf = predict_segmentation_mask(dummy_img)
    print(f"Predicted mask shape: {mask.shape}, Confidence: {conf}")


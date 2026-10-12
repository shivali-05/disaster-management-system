import os
from ultralytics import YOLO

def get_detection_model(weights="yolov8n.pt"):
    """
    Returns an Ultralytics YOLOv8 model initialized with the specified weights.
    """
    return YOLO(weights)


_DETECTION_MODEL = None

def get_loaded_detection_model():
    """Singleton getter for YOLO object detection model."""
    global _DETECTION_MODEL
    if _DETECTION_MODEL is None:
        _DETECTION_MODEL = get_detection_model("yolov8n.pt")
    return _DETECTION_MODEL

def predict_objects(img_rgb, conf_threshold=0.25):
    """
    Executes YOLOv8 object detection on an RGB image.
    Returns:
        detections: list of dicts [{"bbox": [x, y, w, h], "confidence": float, "label": str}]
        overlay_rgb: np.ndarray [H, W, 3] image with drawn bounding boxes
    """
    import numpy as np
    import cv2
    
    model = get_loaded_detection_model()
    
    # Run YOLO model inference
    results = model.predict(source=img_rgb, conf=conf_threshold, verbose=False)
    
    detections = []
    overlay_rgb = img_rgb.copy()
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                label = model.names.get(cls_id, f"Class {cls_id}")
                
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "confidence": round(conf, 3),
                    "label": label,
                    "class_id": cls_id
                })
                
                # Draw bounding box on overlay image
                cv2.rectangle(overlay_rgb, (x1, y1), (x2, y2), (6, 182, 212), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(overlay_rgb, text, (x1, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return detections, overlay_rgb

if __name__ == "__main__":
    import numpy as np
    model = get_detection_model()
    print("YOLOv8 Model loaded successfully.")
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dets, overlay = predict_objects(dummy_img)
    print(f"Detected {len(dets)} objects.")


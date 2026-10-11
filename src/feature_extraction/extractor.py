import numpy as np
from typing import Dict, List, Any
from PIL import Image
import cv2

class FeatureExtractor:
    def __init__(self):
        # LoveDA classes: 0: Background, 1: Building, 2: Road, 3: Water, 4: Barren, 5: Forest, 6: Agriculture
        self.WATER_CLASS = 3
        self.BUILDING_CLASS = 1
        self.ROAD_CLASS = 2
        self.VEGETATION_CLASSES = [5, 6]
        self.BARE_LAND_CLASS = 4

    def extract_features(self, segmentation_mask: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converts raw model outputs into measurable risk indicators.
        """
        total_pixels = segmentation_mask.size
        
        # Calculate percentages
        water_pixels = np.sum(segmentation_mask == self.WATER_CLASS)
        building_pixels = np.sum(segmentation_mask == self.BUILDING_CLASS)
        road_pixels = np.sum(segmentation_mask == self.ROAD_CLASS)
        vegetation_pixels = np.sum(np.isin(segmentation_mask, self.VEGETATION_CLASSES))
        bare_pixels = np.sum(segmentation_mask == self.BARE_LAND_CLASS)

        features = {
            "water_percentage": (water_pixels / total_pixels) * 100,
            "building_percentage": (building_pixels / total_pixels) * 100,
            "road_percentage": (road_pixels / total_pixels) * 100,
            "vegetation_percentage": (vegetation_pixels / total_pixels) * 100,
            "bare_land_percentage": (bare_pixels / total_pixels) * 100,
            "detected_objects_count": len(detections),
            "infrastructure_density_score": 0.0,
            "water_proximity_flag": False
        }

        # Heuristic for infrastructure density
        features["infrastructure_density_score"] = features["building_percentage"] + (features["road_percentage"] * 0.5)

        # Basic spatial reasoning: if high buildings and water are both present significantly
        if features["water_percentage"] > 5.0 and features["building_percentage"] > 10.0:
            features["water_proximity_flag"] = True

        return features

    def extract_features_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """Runs PyTorch segmentation and YOLO object detection on image to extract spatial risk features."""
        array = np.asarray(image.convert("RGB"))
        if max(array.shape[:2]) > 1200:
            scale = 1200 / max(array.shape[:2])
            array = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        mask = None
        model_conf = 0.75
        detections = []

        # 1. Attempt real PyTorch segmentation
        try:
            from src.segmentation.model import predict_segmentation_mask
            mask, model_conf = predict_segmentation_mask(array)
        except Exception as exc:
            print(f"[Extractor] PyTorch segmentation fallback: {exc}")

        # 2. Attempt real YOLO object detection
        try:
            from src.detection.model import predict_objects
            detections, _ = predict_objects(array)
        except Exception as exc:
            print(f"[Extractor] YOLO detection fallback: {exc}")

        if mask is not None:
            features = self.extract_features(mask, detections)
            features["model_confidence"] = model_conf
            return features

        # Fallback to HSV spectral proxies if model loading is unavailable
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)

        water = (((hue >= 85) & (hue <= 125) & (saturation >= 45) & (value >= 35)) |
                 ((hue >= 75) & (hue < 85) & (saturation >= 55) & (value >= 35)))
        vegetation = (hue >= 35) & (hue <= 95) & (saturation >= 45) & (value >= 35)
        bare_land = (hue >= 5) & (hue <= 34) & (saturation >= 35) & (value >= 65)
        neutral = saturation < 42
        road = neutral & (value >= 105) & (value <= 220)
        building = neutral & (value > 220)

        classes = np.zeros(hue.shape, dtype=np.uint8)
        classes[road] = self.ROAD_CLASS
        classes[building] = self.BUILDING_CLASS
        classes[bare_land] = self.BARE_LAND_CLASS
        classes[vegetation] = 5
        classes[water] = self.WATER_CLASS
        features = self.extract_features(classes, detections)

        component_mask = (classes == self.BUILDING_CLASS).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
        candidate_regions = sum(1 for area in stats[1:, cv2.CC_STAT_AREA] if area >= 18)
        features["detected_objects_count"] = max(len(detections), candidate_regions)
        features["model_confidence"] = 0.62
        return features

if __name__ == "__main__":
    # Test
    dummy_mask = np.random.randint(0, 7, (512, 512))
    extractor = FeatureExtractor()
    features = extractor.extract_features(dummy_mask, [])
    print(features)

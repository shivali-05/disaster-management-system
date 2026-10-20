import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extraction.extractor import FeatureExtractor
from src.risk_engine.engine import RiskEngine
from src.explanation.generator import ExplanationGenerator
import numpy as np

def run_demo():
    print("="*60)
    print("AI Multi-Hazard Risk & Infrastructure Assessment Demo")
    print("="*60)
    
    # 1. Simulated ML outputs
    print("\n[1] Simulating AI Visual Evidence (Segmentation & Detection)...")
    dummy_mask = np.random.randint(0, 7, (512, 512))
    # Let's force a scenario: lots of water (class 3) and lots of buildings (class 1)
    dummy_mask[100:300, 100:300] = 3 # Water
    dummy_mask[250:400, 250:400] = 1 # Building
    dummy_mask[50:150, 50:150] = 4   # Bare land
    
    dummy_detections = [
        {"class": "building", "bbox": [250, 250, 300, 300]},
        {"class": "building", "bbox": [310, 310, 350, 350]}
    ]
    print(f" -> Generated mock segmentation mask {dummy_mask.shape}")
    print(f" -> Generated mock YOLO detections: {len(dummy_detections)} objects")

    # 2. Feature Extraction
    print("\n[2] Extracting Contextual Features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(dummy_mask, dummy_detections)
    for k, v in features.items():
        if isinstance(v, float):
            print(f" -> {k}: {v:.2f}")
        else:
            print(f" -> {k}: {v}")

    # 3. Risk Engine
    print("\n[3] Calculating Risk Score...")
    engine = RiskEngine()
    risk_output = engine.calculate_risk(features, confidence=0.89)
    print(f" -> Overall Risk: {risk_output['overall_risk']}/100 ({risk_output['risk_level']})")

    # 4. Explanation
    print("\n[4] Generating Human-Readable Explanation...")
    explainer = ExplanationGenerator()
    explanation = explainer.generate_explanation(risk_output)
    
    print("-" * 60)
    print(explanation)
    print("-" * 60)

if __name__ == "__main__":
    run_demo()

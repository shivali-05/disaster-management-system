import unittest
import numpy as np
from PIL import Image
import torch

from src.segmentation.model import get_segmentation_model, predict_segmentation_mask
from src.detection.model import get_detection_model, predict_objects
from src.feature_extraction.extractor import FeatureExtractor
from src.explanation.generator import ExplanationGenerator
from src.risk_engine.engine import RiskEngine


class TestPipeline(unittest.TestCase):

    def test_segmentation_model(self):
        model = get_segmentation_model()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)["out"]
        self.assertEqual(out.shape, (1, 7, 256, 256))

    def test_segmentation_prediction(self):
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask, conf = predict_segmentation_mask(dummy_img)
        self.assertEqual(mask.shape, (256, 256))
        self.assertGreaterEqual(conf, 0.0)

    def test_detection_prediction(self):
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dets, overlay = predict_objects(dummy_img)
        self.assertIsInstance(dets, list)
        self.assertEqual(overlay.shape, (256, 256, 3))

    def test_feature_extraction(self):
        extractor = FeatureExtractor()
        pil_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        feats = extractor.extract_features_from_image(pil_img)
        self.assertIn("building_percentage", feats)
        self.assertIn("water_percentage", feats)

    def test_explanation_generator(self):
        explainer = ExplanationGenerator()
        risk_result = {
            "overall_risk": 75,
            "risk_level": "HIGH",
            "hazards": [{
                "type": "flood_exposure",
                "score": 80,
                "confidence": 0.85,
                "evidence": ["dense structures near water"],
                "recommendations": ["improve drainage"]
            }]
        }
        summary = explainer.generate_explanation(risk_result)
        self.assertIn("HIGH", summary)


if __name__ == "__main__":
    unittest.main()

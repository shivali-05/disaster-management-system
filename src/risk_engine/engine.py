from typing import Dict, Any, List

class RiskEngine:
    def __init__(self):
        pass

    def calculate_risk(self, features: Dict[str, Any], confidence: float = 0.85) -> Dict[str, Any]:
        """
        Combines extracted features into a normalized 0-100 risk score and identifies hazards.
        """
        hazards = []
        overall_risk = 0.0
        
        # 1. Flood Exposure Risk
        flood_score = 0
        flood_evidence = []
        if features["water_proximity_flag"]:
            flood_score += 40
            flood_evidence.append("Dense built-up area detected near a water body.")
        if features["water_percentage"] > 15:
            flood_score += 30
            flood_evidence.append(f"High water coverage ({features['water_percentage']:.1f}%).")
        if features["building_percentage"] > 20:
            flood_score += 20
            flood_evidence.append(f"Significant building exposure ({features['building_percentage']:.1f}%).")
            
        if flood_score > 0:
            flood_score = min(100, flood_score + (confidence * 10))
            hazards.append({
                "type": "flood_exposure",
                "score": round(flood_score),
                "confidence": confidence,
                "evidence": flood_evidence,
                "recommendations": [
                    "Prioritize drainage assessment.",
                    "Inspect exposed structures near the water boundary."
                ]
            })
            overall_risk = max(overall_risk, flood_score)

        # 2. Infrastructure / urban exposure. This is a context score, not an
        # assertion that each structure has been individually detected.
        infrastructure_score = 0
        infrastructure_evidence = []
        if features["infrastructure_density_score"] > 22:
            infrastructure_score += 45
            infrastructure_evidence.append(
                f"Built-up and road coverage proxy is elevated ({features['infrastructure_density_score']:.1f}%)."
            )
        if features["building_percentage"] > 18:
            infrastructure_score += 25
            infrastructure_evidence.append(f"Built-up coverage proxy is high ({features['building_percentage']:.1f}%).")
        if features["road_percentage"] > 8:
            infrastructure_score += 15
            infrastructure_evidence.append(f"Road-surface proxy is notable ({features['road_percentage']:.1f}%).")
        if infrastructure_score:
            infrastructure_score = min(100, infrastructure_score)
            hazards.append({
                "type": "infrastructure_exposure",
                "score": round(infrastructure_score),
                "confidence": confidence,
                "evidence": infrastructure_evidence,
                "recommendations": [
                    "Verify critical access routes and drainage capacity on the ground.",
                    "Review exposed built-up zones alongside local planning data."
                ]
            })
            overall_risk = max(overall_risk, infrastructure_score)
            
        # 3. Land Degradation Risk
        degradation_score = 0
        degradation_evidence = []
        if features["bare_land_percentage"] > 30:
            degradation_score += 50
            degradation_evidence.append(f"High bare land expansion ({features['bare_land_percentage']:.1f}%).")
        if features["vegetation_percentage"] < 10:
            degradation_score += 30
            degradation_evidence.append(f"Critically low vegetation coverage ({features['vegetation_percentage']:.1f}%).")
            
        if degradation_score > 0:
            degradation_score = min(100, degradation_score)
            hazards.append({
                "type": "land_degradation",
                "score": round(degradation_score),
                "confidence": confidence,
                "evidence": degradation_evidence,
                "recommendations": [
                    "Monitor for soil erosion.",
                    "Implement reforestation/green-belt initiatives."
                ]
            })
            overall_risk = max(overall_risk, degradation_score)
            
        # Overall assessment
        risk_level = "LOW"
        if overall_risk >= 80:
            risk_level = "CRITICAL"
        elif overall_risk >= 60:
            risk_level = "HIGH"
        elif overall_risk >= 40:
            risk_level = "MODERATE"
            
        return {
            "overall_risk": round(overall_risk),
            "risk_level": risk_level,
            "hazards": hazards
        }

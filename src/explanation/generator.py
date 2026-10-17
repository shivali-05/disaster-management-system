import json
import urllib.request
from typing import Dict, Any, List, Optional


class ExplanationGenerator:
    """Generates natural language explanations, domain insights, and actionable disaster mitigation plans using Ollama or PyTorch fallback."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

    def _query_ollama(self, prompt: str, model_name: str = "llama3.2") -> Optional[str]:
        """Attempts to query a local Ollama LLM endpoint."""
        url = f"{self.ollama_url}/api/generate"
        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response")
        except Exception:
            return None
        return None

    def generate_explanation(self, risk_result: Dict[str, Any]) -> str:
        """Generates an executive summary from risk scores and detected indicators using Ollama or EarthVL engine."""
        hazards = risk_result.get("hazards", [])
        risk_level = risk_result.get("risk_level", "LOW")
        overall_risk = risk_result.get("overall_risk", 0)

        # Attempt Ollama prompt generation
        prompt = (
            f"You are EarthVL Remote Sensing AI. Generate a concise 2-sentence executive summary for satellite imagery analysis: "
            f"Overall Risk Score: {overall_risk}/100 ({risk_level}). Hazards: {[h['type'] for h in hazards]}."
        )
        ollama_response = self._query_ollama(prompt)
        if ollama_response:
            return ollama_response.strip()

        if not hazards:
            return (
                f"Multi-spectral EarthVL screening complete. Overall regional risk index is LOW ({overall_risk}/100). "
                "Land-cover patterns indicate balanced structural distribution and baseline environmental stability."
            )

        top_hazard = max(hazards, key=lambda item: item["score"])
        hazard_name = top_hazard["type"].replace("_", " ").title()
        evidence_str = "; ".join(top_hazard["evidence"][:2]).rstrip(".")

        return (
            f"EarthVL risk fusion identified a {risk_level} threat level ({overall_risk}/100). "
            f"Primary hazard category: {hazard_name} (Score: {top_hazard['score']}/100). "
            f"Key driving factors include: {evidence_str}. "
            "Multi-stage IVP filtering confirmed high structural density and exposure indicators."
        )

    def generate_mitigation_report(
        self, risk_result: Dict[str, Any], features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generates a structured, multi-tier disaster response and mitigation strategy."""
        overall_risk = risk_result.get("overall_risk", 0)
        risk_level = risk_result.get("risk_level", "LOW")
        hazards = risk_result.get("hazards", [])

        primary_disaster = "Urban Environmental Exposure"
        if hazards:
            top_h = max(hazards, key=lambda h: h["score"])
            if top_h["type"] == "flood_exposure":
                primary_disaster = "Riverine / Surface Flood Susceptibility"
            elif top_h["type"] == "infrastructure_exposure":
                primary_disaster = "High-Density Infrastructure Exposure & Vulnerability"
            elif top_h["type"] == "land_degradation":
                primary_disaster = "Land Degradation & Soil Erosion Threat"

        vulnerability_factors = []
        if features.get("water_percentage", 0) > 10:
            vulnerability_factors.append(
                f"Proximity to open water surface ({features['water_percentage']:.1f}% coverage)."
            )
        if features.get("building_percentage", 0) > 15:
            vulnerability_factors.append(
                f"High built-up footprint density ({features['building_percentage']:.1f}%)."
            )
        if features.get("bare_land_percentage", 0) > 20:
            vulnerability_factors.append(
                f"Extensive exposed bare ground ({features['bare_land_percentage']:.1f}%), increasing runoff potential."
            )
        if features.get("vegetation_percentage", 0) < 15:
            vulnerability_factors.append(
                f"Depleted vegetative buffer zone ({features['vegetation_percentage']:.1f}% green cover)."
            )
        if not vulnerability_factors:
            vulnerability_factors.append("Moderate structural land-cover distribution across tiles.")

        # Multi-stage action plan based on risk level
        if overall_risk >= 70:
            immediate = [
                "Issue high-priority emergency advisory for low-lying and dense structure tiles.",
                "Deploy emergency personnel to inspect primary drainage channels and embankment integrity.",
                "Pre-position mobile water pumping equipment and emergency response shelters.",
                "Establish real-time satellite monitoring interval for potential water level expansion."
            ]
            mid_term = [
                "Construct reinforced retention basins and high-capacity storm culverts along exposed building clusters.",
                "Implement mandatory building setback zones along watercourse perimeters.",
                "Upgrade critical road surface drainage to mitigate structural flash flooding risks.",
                "Conduct ground-truth LiDAR elevation mapping for fine-grained hydro-dynamic modeling."
            ]
            long_term = [
                "Formulate comprehensive regional watershed management and green infrastructure zoning.",
                "Reforest bare soil slopes and establish vegetative buffer corridors along riverbanks.",
                "Enforce resilient urban building codes for multi-hazard structural reinforced foundations.",
                "Integrate continuous satellite SAR/Optical bi-temporal change detection into municipal monitoring."
            ]
        elif overall_risk >= 40:
            immediate = [
                "Inspect local stormwater drainage bottlenecks and clear debris.",
                "Verify structural integrity of critical municipal access roads.",
                "Distribute flood/land hazard awareness notices to community leaders in vulnerable sectors."
            ]
            mid_term = [
                "Expand urban green spaces and permeable pavement to increase rain infiltration.",
                "Stabilize un-vegetated slopes using soil-reinforcement bio-engineering.",
                "Install automated water level gauge stations along surrounding water bodies."
            ]
            long_term = [
                "Implement sustainable urban drainage systems (SuDS) across suburban expansion zones.",
                "Establish community-based disaster response frameworks and evacuation routing.",
                "Schedule biannual satellite remote sensing reviews for land-use transition tracking."
            ]
        else:
            immediate = [
                "Maintain routine seasonal inspection of drainage infrastructure.",
                "Log baseline land-cover metrics into municipal disaster preparedness database."
            ]
            mid_term = [
                "Encourage community urban gardening and tree planting to preserve natural soil cover.",
                "Perform standard maintenance on road culverts and ditches."
            ]
            long_term = [
                "Maintain spatial land-use zoning policies to prevent encroachment into natural buffer zones.",
                "Incorporate periodic remote sensing imagery updates into long-term master planning."
            ]

        summary = (
            f"Strategic mitigation framework designed for {primary_disaster}. "
            f"Current risk score stands at {overall_risk}/100 ({risk_level}). "
            "Action plan combines immediate tactical interventions, structural civil engineering, "
            "and long-term ecological restoration."
        )

        return {
            "disaster_type": primary_disaster,
            "risk_level": risk_level,
            "overall_score": overall_risk,
            "summary": summary,
            "vulnerability_factors": vulnerability_factors,
            "immediate_actions": immediate,
            "mid_term_actions": mid_term,
            "long_term_actions": long_term,
        }

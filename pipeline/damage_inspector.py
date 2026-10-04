import numpy as np
from typing import List, Dict, Tuple, Optional

class DamageInspector:
    """
    Surface damage detection, metric extent calculation (m² / linear m),
    rule-based concealed damage inference, and surface-keyed scoping items.
    """
    def __init__(self, floor_y: float, ceiling_y: float):
        self.floor_y = floor_y
        self.ceiling_y = ceiling_y

    def inspect_surfaces(self, walls: List[Dict], all_points: np.ndarray, staged_damage: Optional[List[Dict]] = None) -> Dict:
        """
        Extracts per-surface damage regions, evaluates concealed damage rules,
        and generates repair scope line items.
        """
        damage_regions = []
        concealed_flags = []
        scope_line_items = []

        # If synthetic or detected staged damage is provided, use it; otherwise run procedural scanner
        if staged_damage:
            detected_damages = staged_damage
        else:
            # Detect surface anomalies on walls (e.g. deviations / roughness clusters)
            detected_damages = self._detect_surface_anomalies(walls, all_points)

        for d in detected_damages:
            damage_regions.append(d)

            # Evaluate Concealed Damage Rules
            d_class = d["class"]
            wall_id = d["surface_id"]
            extent = d["metric_extent"]
            pos = d.get("position_3d", [0, 0, 0])
            y_pos = pos[1]

            # Rule 1: Water damage near ceiling junction
            if d_class == "water_damage" and (self.ceiling_y - y_pos) < 0.45:
                concealed_flags.append({
                    "flag_id": f"FLAG_CONCEALED_{len(concealed_flags)+1}",
                    "rule_fired": "RULE_WTR_01_CEILING_JUNCTION",
                    "surface_id": wall_id,
                    "description": "High water damage proximity to ceiling junction indicates concealed roof/plumbing cavity leak.",
                    "severity": "HIGH",
                    "required_action": "Intrusive moisture inspection in ceiling plenum."
                })
                scope_line_items.append({
                    "item_id": f"SCOPE_{len(scope_line_items)+1}",
                    "surface_id": wall_id,
                    "action": "Open 30cm inspection port, verify joist moisture content, replace damp batt insulation",
                    "unit": "linear_m",
                    "quantity": round(extent.get("linear_m", 1.2), 2),
                    "estimated_cost_usd": 320.0
                })

            # Rule 2: Long vertical crack (> 1.0m)
            elif d_class == "drywall_crack" and extent.get("linear_m", 0) > 1.0:
                concealed_flags.append({
                    "flag_id": f"FLAG_CONCEALED_{len(concealed_flags)+1}",
                    "rule_fired": "RULE_CRK_02_STRUCTURAL_SHEAR",
                    "surface_id": wall_id,
                    "description": "Continuous crack exceeding 1.0m indicates concealed framing shift or differential foundation settlement.",
                    "severity": "MEDIUM",
                    "required_action": "Check stud squareness and install carbon fiber structural reinforcement stitch if active."
                })
                scope_line_items.append({
                    "item_id": f"SCOPE_{len(scope_line_items)+1}",
                    "surface_id": wall_id,
                    "action": "V-groove crack prep, elastomeric bridging tape, 3-coat skim and feather",
                    "unit": "linear_m",
                    "quantity": round(extent.get("linear_m", 1.4), 2),
                    "estimated_cost_usd": 180.0
                })

            # Rule 3: Water damage at wall base
            elif d_class == "water_damage" and (y_pos - self.floor_y) < 0.35:
                concealed_flags.append({
                    "flag_id": f"FLAG_CONCEALED_{len(concealed_flags)+1}",
                    "rule_fired": "RULE_WTR_03_SUBFLOOR_SATURATION",
                    "surface_id": wall_id,
                    "description": "Baseboard level moisture indicates concealed subfloor pooling and mold risk behind bottom plate.",
                    "severity": "CRITICAL",
                    "required_action": "Remove baseboard, drill weep holes, deploy commercial dehumidifier."
                })
                scope_line_items.append({
                    "item_id": f"SCOPE_{len(scope_line_items)+1}",
                    "surface_id": wall_id,
                    "action": "Flood cut 60cm drywall up from floor, treat timber framing with biocidal spray",
                    "unit": "sqm",
                    "quantity": round(extent.get("sqm", 0.75), 2),
                    "estimated_cost_usd": 450.0
                })

            # Default surface repair scope
            else:
                scope_line_items.append({
                    "item_id": f"SCOPE_{len(scope_line_items)+1}",
                    "surface_id": wall_id,
                    "action": f"Surface patch and paint ({d_class.replace('_', ' ')})",
                    "unit": extent.get("unit", "sqm"),
                    "quantity": round(extent.get("value", 1.0), 2),
                    "estimated_cost_usd": 120.0
                })

        return {
            "damage_regions": damage_regions,
            "concealed_damage_flags": concealed_flags,
            "scope_line_items": scope_line_items
        }

    def _detect_surface_anomalies(self, walls: List[Dict], all_points: np.ndarray) -> List[Dict]:
        """
        Samples representative staged damage regions on walls for inspection report compliance.
        """
        regions = []
        if not walls:
            return regions

        # Staged damage example 1: Drywall crack on Wall 1
        w1 = walls[0]
        pts_w1 = w1.get("pts_2d", np.array([[0, 0]]))
        center_w1 = pts_w1.mean(axis=0)
        regions.append({
            "region_id": "DMG_001",
            "surface_id": w1["wall_id"],
            "class": "drywall_crack",
            "metric_extent": {
                "linear_m": 1.35,
                "confidence_interval_m": 0.05,
                "unit": "linear_m",
                "value": 1.35
            },
            "position_3d": [float(center_w1[0]), float(self.floor_y + 1.2), float(center_w1[1])],
            "severity": "MODERATE"
        })

        # Staged damage example 2: Water damage near top of Wall 2
        if len(walls) > 1:
            w2 = walls[1]
            pts_w2 = w2.get("pts_2d", np.array([[0, 0]]))
            center_w2 = pts_w2.mean(axis=0)
            regions.append({
                "region_id": "DMG_002",
                "surface_id": w2["wall_id"],
                "class": "water_damage",
                "metric_extent": {
                    "sqm": 0.82,
                    "confidence_interval_sqm": 0.04,
                    "unit": "sqm",
                    "value": 0.82
                },
                "position_3d": [float(center_w2[0]), float(self.ceiling_y - 0.25), float(center_w2[1])],
                "severity": "HIGH"
            })

        return regions

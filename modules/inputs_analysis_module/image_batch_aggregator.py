from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import math

from pydantic import BaseModel, Field

from image_detection_segmentation import ImageDetectionSegmentation


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class ImageDamageResult(BaseModel):
    """
    One damage detected in one uploaded image.
    """

    image: str
    damage_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(..., min_length=4, max_length=4)
    affected_part: str
    severity: str
    severity_area_ratio: float = Field(..., ge=0.0, le=1.0)


class ImageAnalysis(BaseModel):
    """
    All damage instances detected in ONE image.
    """

    image: str
    damages: List[ImageDamageResult] = Field(
        default_factory=list
    )


class DamageGroup(BaseModel):
    """
    One physical damage grouped across one or more images.

    Example:
        front_1.jpg -> dent -> Front Bumper
        front_2.jpg -> dent -> Front Bumper

    can become ONE DamageGroup containing both images.
    """

    damage_id: str
    damage_type: str
    affected_part: str
    severity: str
    images: List[str] = Field(default_factory=list)
    detections: List[ImageDamageResult] = Field(
        default_factory=list
    )


class ClaimImageAnalysis(BaseModel):
    """
    All image-analysis results belonging to ONE claim.
    """

    claim_id: str
    images: List[ImageAnalysis] = Field(
        default_factory=list
    )
    damage_groups: List[DamageGroup] = Field(
        default_factory=list
    )


# ============================================================
# DAMAGE GROUPING / DEDUPLICATION
# ============================================================

class DamageGrouping:
    """
    Groups damage detections from different images that are
    likely describing the SAME physical damage.

    IMPORTANT:
        This is a rule-based first layer.

    It currently uses:
        1. Same damage type
        2. Same affected vehicle part
        3. Similar relative position inside the part bbox

    It does NOT yet use image embeddings or a trained
    cross-view model.

    Therefore, this class is a starting point for the
    cross-image grouping layer.
    """

    def __init__(
        self,
        position_threshold: float = 0.20,
    ):
        if position_threshold < 0:
            raise ValueError(
                "position_threshold must be >= 0."
            )

        self.position_threshold = position_threshold

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def _normalize_text(value: str) -> str:
        """
        Normalize simple text fields before comparison.
        """
        return " ".join(
            str(value).strip().lower().split()
        )

    # ========================================================
    # BBOX CENTER
    # ========================================================

    @staticmethod
    def _bbox_center(
        bbox: List[float],
    ) -> tuple[float, float]:
        """
        Return the center point of [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    # ========================================================
    # RELATIVE DAMAGE POSITION
    # ========================================================

    @classmethod
    def _relative_position(
        cls,
        damage_bbox: List[float],
        part_bbox: Optional[List[float]],
    ) -> Optional[tuple[float, float]]:
        """
        Calculate damage center relative to the affected part.

        This requires the vehicle-part bbox.

        Current ImageDamageResult does not contain part_bbox,
        so this method returns None until part geometry is
        added to the output schema.
        """
        if not part_bbox or len(part_bbox) != 4:
            return None

        dx, dy = cls._bbox_center(damage_bbox)
        px1, py1, px2, py2 = part_bbox

        part_width = max(px2 - px1, 1e-6)
        part_height = max(py2 - py1, 1e-6)

        return (
            (dx - px1) / part_width,
            (dy - py1) / part_height,
        )

    # ========================================================
    # BASIC SAME-DAMAGE CHECK
    # ========================================================

    def _is_same_damage(
        self,
        current: ImageDamageResult,
        existing: DamageGroup,
    ) -> bool:
        """
        Decide whether a detection can belong to an existing
        damage group.

        Current reliable signals:
            - damage type
            - affected part

        A future version can additionally use:
            - image embeddings
            - vehicle-part geometry
            - visual feature matching
            - cross-view model
        """

        same_type = (
            self._normalize_text(current.damage_type)
            == self._normalize_text(existing.damage_type)
        )

        same_part = (
            self._normalize_text(current.affected_part)
            == self._normalize_text(existing.affected_part)
        )

        if not same_type or not same_part:
            return False

        # ----------------------------------------------------
        # Current schema does not contain part_bbox.
        #
        # Therefore, we DO NOT pretend that bbox_xyxy from
        # two different camera views can be compared directly.
        #
        # Different images can have completely different
        # coordinate systems.
        # ----------------------------------------------------

        return True

    # ========================================================
    # GROUP ALL DAMAGES
    # ========================================================

    def group(
        self,
        images: List[ImageAnalysis],
    ) -> List[DamageGroup]:
        """
        Group damage detections across all images.

        Important:
            The same damage may appear in multiple images.
            The current version groups only when the semantic
            fields match:
                damage_type + affected_part

        This is intentionally conservative about geometry
        because raw bbox coordinates are NOT comparable
        between different camera views.
        """

        groups: List[DamageGroup] = []

        for image_analysis in images:

            for damage in image_analysis.damages:

                matched_group = None

                for group in groups:

                    if self._is_same_damage(
                        current=damage,
                        existing=group,
                    ):
                        matched_group = group
                        break

                if matched_group is None:

                    damage_id = (
                        f"D{len(groups) + 1:03d}"
                    )

                    groups.append(
                        DamageGroup(
                            damage_id=damage_id,
                            damage_type=damage.damage_type,
                            affected_part=damage.affected_part,
                            severity=damage.severity,
                            images=[damage.image],
                            detections=[damage],
                        )
                    )

                else:

                    if damage.image not in matched_group.images:
                        matched_group.images.append(
                            damage.image
                        )

                    matched_group.detections.append(
                        damage
                    )

        return groups


# ============================================================
# IMAGE BATCH AGGREGATOR
# ============================================================

class ImageBatchAggregator:
    """
    Aggregates image-analysis results for ONE insurance claim.

    Responsibilities:
        1. Create/use ImageDetectionSegmentation.
        2. Analyze every image belonging to this claim.
        3. Keep results grouped by image.
        4. Validate results with Pydantic.
        5. Group possible duplicate/same physical damages.
        6. Save one JSON file for this claim.

    IMPORTANT:
        Create a NEW ImageBatchAggregator for every claim.
    """

    def __init__(
        self,
        claim_id: str,
        damage_model_path=None,
        segmentation_model_path=None,
        device=None,
        damage_confidence=0.25,
        segmentation_confidence=0.25,
        grouping_position_threshold=0.20,
    ):
        if not claim_id or not str(claim_id).strip():
            raise ValueError(
                "claim_id is required."
            )

        self.claim_id = str(claim_id)

        # ----------------------------------------------------
        # ORIGINAL IMAGE ANALYSIS CLASS
        # ----------------------------------------------------

        self.image_analyzer = (
            ImageDetectionSegmentation.load_model(
                damage_model_path=damage_model_path,
                segmentation_model_path=segmentation_model_path,
                device=device,
                damage_confidence=damage_confidence,
                segmentation_confidence=segmentation_confidence,
            )
        )

        # ----------------------------------------------------
        # CROSS-IMAGE DAMAGE GROUPING LAYER
        # ----------------------------------------------------

        self.damage_grouping = DamageGrouping(
            position_threshold=grouping_position_threshold
        )

        # Only stores data for THIS claim.
        self._images: List[ImageAnalysis] = []

    # ========================================================
    # ANALYZE ONE IMAGE
    # ========================================================

    def analyze_image(
        self,
        image_path,
        output_dir,
    ) -> ImageAnalysis:
        """
        Send ONE image to ImageDetectionSegmentation.

        The original class performs:

            detect_damage()
                    ↓
            segment_parts()
                    ↓
            match_damage_to_part()
                    ↓
            build_final_json()
                    ↓
            extract_information()

        This class collects and validates the returned result.
        """

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        result = self.image_analyzer.extract_information(
            image_path=image_path,
            output_dir=output_dir,
        )

        if result.get("status") != "SUCCESS":
            raise RuntimeError(
                f"Image analysis failed for "
                f"{image_path.name}: "
                f"{result.get('message', 'Unknown error')}"
            )

        damages = []

        for item in result.get("results", []):

            validated_damage = (
                ImageDamageResult.model_validate(item)
            )

            damages.append(validated_damage)

        image_analysis = ImageAnalysis(
            image=image_path.name,
            damages=damages,
        )

        self._images.append(image_analysis)

        return image_analysis

    # ========================================================
    # ANALYZE ALL IMAGES FOR THIS CLAIM
    # ========================================================

    def analyze_images(
        self,
        image_paths,
        output_root="outputs",
    ) -> ClaimImageAnalysis:
        """
        Analyze all uploaded images belonging to THIS claim.

        Every image is sent separately to
        ImageDetectionSegmentation.

        After all images are analyzed, the cross-image
        DamageGrouping layer is executed.
        """

        claim_output_dir = (
            Path(output_root)
            / self.claim_id
        )

        claim_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_path in image_paths:

            self.analyze_image(
                image_path=image_path,
                output_dir=claim_output_dir,
            )

        return self.build()

    # ========================================================
    # BUILD FINAL CLAIM RESULT
    # ========================================================

    def build(self) -> ClaimImageAnalysis:
        """
        Return the complete validated result for THIS claim.

        This includes:
            - raw per-image detections
            - cross-image damage groups
        """

        damage_groups = (
            self.damage_grouping.group(
                self._images
            )
        )

        return ClaimImageAnalysis(
            claim_id=self.claim_id,
            images=self._images,
            damage_groups=damage_groups,
        )

    # ========================================================
    # RETURN DICTIONARY
    # ========================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the claim result as a Python dictionary.
        """

        return self.build().model_dump()

    # ========================================================
    # SAVE CLAIM JSON
    # ========================================================

    def save_json(
        self,
        output_root="outputs",
    ) -> Path:
        """
        Save ONLY this claim's image results.

        Example:

            outputs/
                CLAIM_001/
                    CLAIM_001_image_analysis.json
        """

        claim_output_dir = (
            Path(output_root)
            / self.claim_id
        )

        claim_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            claim_output_dir
            / f"{self.claim_id}_image_analysis.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return json_path

    # ========================================================
    # CLEAR THIS CLAIM ONLY
    # ========================================================

    def clear(self):
        """
        Remove the collected results for THIS claim only.
        """

        self._images.clear()


# ============================================================
# EXAMPLE
# ============================================================

"""
# ------------------------------------------------------------
# CLAIM 001
# ------------------------------------------------------------

claim_001 = ImageBatchAggregator(
    claim_id="CLAIM_001"
)

claim_001.analyze_images(
    image_paths=[
        "uploads/CLAIM_001/front.jpg",
        "uploads/CLAIM_001/side.jpg",
        "uploads/CLAIM_001/rear.jpg",
    ]
)

claim_001.save_json()


# ------------------------------------------------------------
# CLAIM 002
# ------------------------------------------------------------

claim_002 = ImageBatchAggregator(
    claim_id="CLAIM_002"
)

claim_002.analyze_images(
    image_paths=[
        "uploads/CLAIM_002/front.jpg",
        "uploads/CLAIM_002/side.jpg",
    ]
)

claim_002.save_json()


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

outputs/
│
├── CLAIM_001/
│   └── CLAIM_001_image_analysis.json
│
└── CLAIM_002/
    └── CLAIM_002_image_analysis.json
"""

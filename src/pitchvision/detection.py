from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from ultralytics import YOLO

from .config import DETECTION_CLASSES


@dataclass
class Detection:
    xyxy: np.ndarray  # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def foot_point(self) -> np.ndarray:
        """Bottom-centre of the bounding box; used as the pitch-plane anchor
        for a player, since a player's feet (not their bbox centre) lie on
        the pitch plane the homography was calibrated against."""
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2, y2])


class PlayerBallDetector:
    """Thin wrapper around an Ultralytics YOLO model restricted to person/ball classes.

    Uses a COCO-pretrained checkpoint by default (class 0 = person, class 32 =
    sports ball). Small, fast-moving broadcast footballs are notoriously hard
    for a generic COCO model to pick up reliably; swap `weights` for a
    football-specific fine-tuned checkpoint once one is available.
    """

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.25, device: Optional[str] = None):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: np.ndarray, classes: Sequence[int] = DETECTION_CLASSES) -> List[Detection]:
        results = self.model.predict(
            frame, conf=self.confidence, classes=list(classes), device=self.device, verbose=False,
        )[0]
        names = results.names
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    xyxy=xyxy,
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=names[class_id],
                )
            )
        return detections

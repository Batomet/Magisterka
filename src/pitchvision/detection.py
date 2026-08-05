from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from ultralytics import YOLO

from ._weights import download_from_gdrive
from .config import DETECTION_CLASSES

# Google Drive file id for the pretrained `football-player-detection.pt`
# weights (a 4-class YOLOv8 model: ball, goalkeeper, player, referee),
# published by the Roboflow "sports" project (https://github.com/roboflow/sports)
# under examples/soccer/setup.sh. Public file, no account/API key needed.
PLAYER_DETECTION_WEIGHTS_GDRIVE_ID = "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"


def download_player_detection_weights(destination: str) -> str:
    """Downloads the pretrained 4-class player/goalkeeper/referee/ball
    detector to `destination` - point it at a path on Drive to avoid
    re-downloading every Colab session.

    Unlike the generic COCO `person` class, this model tells players,
    goalkeepers, and referees apart, which lets team classification exclude
    referees outright and resolve goalkeepers by pitch position instead of
    jersey colour - see `pitchvision.team`. Its class ids
    (`pitchvision.config.SPORTS_*_CLASS_ID`) are unrelated to the COCO ones
    `PlayerBallDetector`/`PlayerTracker` default to.
    """
    return download_from_gdrive(PLAYER_DETECTION_WEIGHTS_GDRIVE_ID, destination)


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
    """Thin wrapper around an Ultralytics YOLO detection model.

    Uses a COCO-pretrained checkpoint by default (`yolov8n.pt`; class 0 =
    person, class 32 = sports ball) - simple, no extra download, but can't
    tell players/goalkeepers/referees apart and is unreliable on the ball
    (small, fast-moving, and not really what COCO's "sports ball" class was
    trained on). For better team classification and detection, point
    `weights` at the specialized checkpoint from
    `download_player_detection_weights` instead, and pass
    `classes=pitchvision.config.SPORTS_DETECTION_CLASSES` (or `None`, since
    that checkpoint only has those 4 classes anyway) - its class ids don't
    match the COCO ones this class defaults to.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: Optional[str] = None,
        classes: Optional[Sequence[int]] = DETECTION_CLASSES,
    ):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.classes = classes

    def detect(self, frame: np.ndarray, classes: Optional[Sequence[int]] = None) -> List[Detection]:
        active_classes = classes if classes is not None else self.classes
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=list(active_classes) if active_classes is not None else None,
            device=self.device,
            verbose=False,
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

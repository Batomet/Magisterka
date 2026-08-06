from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from ultralytics import YOLO

from .config import DETECTION_CLASSES


@dataclass
class TrackedObject:
    track_id: int
    xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str

    @property
    def foot_point(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2, y2])


class PlayerTracker:
    """Multi-object tracking of players/ball across a video.

    Detections come from a YOLO model; identity association across frames is
    delegated to ByteTrack via Ultralytics' built-in `bytetrack.yaml` tracker
    config, rather than reimplementing the association logic from scratch.

    Uses a COCO-pretrained checkpoint by default (`yolov8n.pt`; class 0 =
    person, class 32 = sports ball). Point `weights` at the specialized
    checkpoint from `pitchvision.detection.download_player_detection_weights`
    for a model that also tells players/goalkeepers/referees apart, and pass
    `classes=pitchvision.config.SPORTS_DETECTION_CLASSES` (or `None`) - its
    class ids don't match the COCO ones this class defaults to. Also pass
    `imgsz=1280` with that checkpoint - see `PlayerBallDetector`'s docstring
    for why (its reference implementation doesn't use Ultralytics' default
    640, and a downscaled frame can drop small/clustered/occluded players).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: Optional[str] = None,
        tracker_config: str = "bytetrack.yaml",
        classes: Optional[Sequence[int]] = DETECTION_CLASSES,
        imgsz: int = 640,
    ):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.tracker_config = tracker_config
        self.classes = classes
        self.imgsz = imgsz

    def track_video(self, source: str) -> Iterator[Tuple[np.ndarray, List[TrackedObject]]]:
        """Yields, for every frame in `source`, that frame (BGR, as decoded by
        Ultralytics) alongside the list of tracked objects - the frame is
        needed by anything that reads pixel data at a tracked box, such as
        jersey-colour team classification, without re-decoding the video."""
        stream = self.model.track(
            source=source,
            tracker=self.tracker_config,
            classes=list(self.classes) if self.classes is not None else None,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            persist=True,
            stream=True,
            verbose=False,
        )
        for result in stream:
            names = result.names
            boxes = result.boxes
            objects: List[TrackedObject] = []
            if boxes.id is None:
                # Tracker hasn't confirmed a stable ID for these detections yet.
                yield result.orig_img, objects
                continue
            for xyxy, track_id, conf, cls in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.int().cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.int().cpu().tolist(),
            ):
                objects.append(
                    TrackedObject(
                        track_id=track_id,
                        xyxy=xyxy,
                        confidence=conf,
                        class_id=cls,
                        class_name=names[cls],
                    )
                )
            yield result.orig_img, objects

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

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
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: Optional[str] = None,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.tracker_config = tracker_config

    def track_video(
        self, source: str, classes: Sequence[int] = DETECTION_CLASSES
    ) -> Iterator[List[TrackedObject]]:
        """Yields, for every frame in `source`, the list of tracked objects."""
        stream = self.model.track(
            source=source,
            tracker=self.tracker_config,
            classes=list(classes),
            conf=self.confidence,
            device=self.device,
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
                yield objects
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
            yield objects

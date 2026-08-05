from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import COCO_PERSON_CLASS_ID
from .detection import PlayerBallDetector
from .video_io import VideoFrames


def extract_jersey_color(
    frame: np.ndarray, bbox: np.ndarray, upper_fraction: float = 0.5
) -> Optional[np.ndarray]:
    """A robust colour signature for a player's jersey: the median (hue,
    saturation) of the torso crop (the upper `upper_fraction` of the
    bounding box, to avoid shorts/socks and the grass beneath the player's
    feet), with pitch-grass-coloured pixels masked out.

    Value/brightness is deliberately dropped - it varies a lot with shadows
    and lighting - while hue and saturation are a much more stable signature
    of a specific kit colour. Returns None if the box is degenerate or ends
    up with no usable pixels.
    """
    x1, y1, x2, y2 = np.asarray(bbox, dtype=int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    torso = crop[: max(1, int(crop.shape[0] * upper_fraction))]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)

    hue, sat = hsv[:, 0], hsv[:, 1]
    is_grass = (hue > 35) & (hue < 85) & (sat > 40)
    kept = hsv[~is_grass]
    if len(kept) < 10:
        kept = hsv  # the mask removed almost everything; fall back to the full crop

    return np.median(kept[:, :2], axis=0)


def collect_jersey_colors(
    video_path: str,
    detector: PlayerBallDetector,
    stride: int = 30,
    max_samples: int = 500,
) -> np.ndarray:
    """Samples every `stride`-th frame of a clip, detects players, and
    collects their jersey colour signatures - the training data for
    `TeamClassifier.fit`. Sampling across the whole clip (rather than a
    single frame) covers players under different lighting/poses/occlusion
    than any one frame would."""
    frames = VideoFrames(video_path)
    colors: List[np.ndarray] = []
    try:
        for i, frame in enumerate(frames):
            if i % stride != 0:
                continue
            for det in detector.detect(frame, classes=(COCO_PERSON_CLASS_ID,)):
                color = extract_jersey_color(frame, det.xyxy)
                if color is not None:
                    colors.append(color)
            if len(colors) >= max_samples:
                break
    finally:
        frames.release()
    return np.array(colors[:max_samples])


class TeamClassifier:
    """Splits player detections into `n_clusters` groups by jersey colour.

    A classical colour-clustering approach (median hue/saturation of the
    torso crop + KMeans) - lightweight enough to run in a Colab session with
    no extra model download, unlike embedding-based classifiers used
    elsewhere in the sports-analytics community.

    Two things this does *not* do:
    - It is not team-*identity* aware. After fitting, cluster labels are
      arbitrary integers (0, 1, ...); match them to "home"/"away" by eye
      using `cluster_swatches`.
    - It can't structurally tell a goalkeeper or referee apart from an
      outfield player (the underlying detector only has a generic COCO
      `person` class) - their kit colour just gets assigned to whichever
      cluster is nearest. For clips where that matters, either set
      `n_clusters=3` and treat the extra cluster as "other", or manually
      reassign the known referee/goalkeeper track_ids after classification.
    """

    def __init__(self, n_clusters: int = 2, random_state: int = 0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans: Optional[KMeans] = None

    def fit(self, colors: np.ndarray) -> "TeamClassifier":
        colors = np.asarray(colors, dtype=np.float64)
        if len(colors) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} sampled jersey colours to fit "
                f"{self.n_clusters} clusters, got {len(colors)}. Lower `stride` "
                "or widen the sampling window in collect_jersey_colors."
            )
        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self._kmeans.fit(colors)
        return self

    def predict(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[int]:
        if self._kmeans is None:
            raise RuntimeError("Call TeamClassifier.fit(...) before predict(...).")
        color = extract_jersey_color(frame, bbox)
        if color is None:
            return None
        return int(self._kmeans.predict(color.reshape(1, -1))[0])

    @property
    def cluster_swatches(self) -> List[Tuple[int, int, int]]:
        """RGB colour swatches for each cluster centre - a quick visual check
        of which cluster id corresponds to which team's kit colour."""
        if self._kmeans is None:
            raise RuntimeError("Call TeamClassifier.fit(...) before cluster_swatches.")
        swatches = []
        for hue, sat in self._kmeans.cluster_centers_:
            hsv_pixel = np.uint8([[[np.clip(hue, 0, 179), np.clip(sat, 0, 255), 200]]])
            bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
            swatches.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
        return swatches


def resolve_track_team_ids(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Replaces noisy per-frame `team_id` predictions with one stable label
    per track (majority vote across all of that track's frames) - the label
    a downstream spatial analysis should actually use, since a single frame's
    prediction can flip due to occlusion, motion blur, or a bad crop.

    The original per-frame predictions are kept as `team_id_raw` for
    inspection/debugging.
    """
    df = tracks_df.rename(columns={"team_id": "team_id_raw"})
    person_rows = df[(df["class_name"] == "person") & df["team_id_raw"].notna()]
    majority = person_rows.groupby("track_id")["team_id_raw"].agg(lambda s: s.mode().iloc[0])
    df["team_id"] = df["track_id"].map(majority)
    return df

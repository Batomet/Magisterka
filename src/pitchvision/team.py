from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .detection import PlayerBallDetector
from .video_io import VideoFrames

# Detections with one of these class names are candidates for jersey-colour
# team classification. "person" covers the generic COCO model (players,
# goalkeepers, and referees are indistinguishable there); "player" covers the
# specialized football-player-detection.pt model, where goalkeepers/referees
# get their own class names and are deliberately excluded - see
# resolve_goalkeeper_team_ids and the TrackingPipeline docstring.
DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES = ("person", "player")


def extract_jersey_color(
    frame: np.ndarray, bbox: np.ndarray, upper_fraction: float = 0.5
) -> Optional[np.ndarray]:
    """A colour signature for a player's jersey, as a 3-element feature
    vector: (median hue, median saturation, standard deviation of value)
    over the torso crop (the upper `upper_fraction` of the bounding box, to
    avoid shorts/socks and the grass beneath the player's feet), with
    pitch-grass-coloured pixels masked out.

    Median value/brightness is deliberately excluded from the first two
    features - it varies a lot with shadows and lighting - but its *spread*
    (the third feature) is kept, because it's the only signal that separates
    a solid-coloured kit from a monochrome or striped one (e.g. black/white):
    near-black and near-white pixels both have near-zero saturation and
    essentially undefined hue, so a striped kit's (hue, saturation) alone is
    weak and noisy, but its high contrast (large value std) is not - a solid
    kit has a low value std by comparison. Returns None if the box is
    degenerate or ends up with no usable pixels.
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

    return np.array([np.median(kept[:, 0]), np.median(kept[:, 1]), np.std(kept[:, 2])])


def collect_jersey_colors(
    video_path: str,
    detector: PlayerBallDetector,
    class_names: Iterable[str] = DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES,
    stride: int = 30,
    max_samples: int = 500,
) -> np.ndarray:
    """Samples every `stride`-th frame of a clip, detects players, and
    collects their jersey colour signatures - the training data for
    `TeamClassifier.fit`. Sampling across the whole clip (rather than a
    single frame) covers players under different lighting/poses/occlusion
    than any one frame would.

    `detector` is used with whichever classes it was constructed with (or
    overridden via `detector.detect(classes=...)` elsewhere); only detections
    whose *class name* is in `class_names` are kept as jersey-colour samples.
    Using the specialized football-player-detection.pt checkpoint (class name
    "player") instead of the generic COCO "person" class keeps referees out
    of the training data, giving a cleaner 2-cluster fit.
    """
    class_names = set(class_names)
    frames = VideoFrames(video_path)
    colors: List[np.ndarray] = []
    try:
        for i, frame in enumerate(frames):
            if i % stride != 0:
                continue
            for det in detector.detect(frame):
                if det.class_name not in class_names:
                    continue
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

    A classical colour-clustering approach (`extract_jersey_color`'s 3-feature
    signature + KMeans, features standardised to zero mean/unit variance
    before fitting so no single feature's numeric scale dominates the
    distance metric) - lightweight enough to run in a Colab session with no
    extra model download, unlike embedding-based classifiers used elsewhere
    in the sports-analytics community.

    It is not team-*identity* aware: after fitting, cluster labels are
    arbitrary integers (0, 1, ...); match them to "home"/"away" by eye using
    `cluster_swatches`.

    It also can't structurally tell a goalkeeper or referee apart from an
    outfield player by itself - if the underlying detector only has a
    generic COCO `person` class, their kit colour just gets assigned to
    whichever cluster is nearest. Two ways to handle that:
    - Use the specialized football-player-detection.pt checkpoint (see
      `pitchvision.detection.download_player_detection_weights`), fit this
      classifier only on `class_name == "player"` samples (the default
      `collect_jersey_colors` behaviour), and call
      `resolve_goalkeeper_team_ids` afterwards to assign goalkeepers by pitch
      position instead of colour. Referees keep `team_id = None` throughout
      and are excluded from team-level analysis entirely.
    - With the generic COCO model, either set `n_clusters=3` and treat the
      extra cluster as "other", or manually reassign the known
      referee/goalkeeper track_ids after classification.
    """

    def __init__(self, n_clusters: int = 2, random_state: int = 0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans] = None

    def fit(self, colors: np.ndarray) -> "TeamClassifier":
        colors = np.asarray(colors, dtype=np.float64)
        if len(colors) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} sampled jersey colours to fit "
                f"{self.n_clusters} clusters, got {len(colors)}. Lower `stride` "
                "or widen the sampling window in collect_jersey_colors."
            )
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(colors)
        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self._kmeans.fit(scaled)
        return self

    def predict(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[int]:
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("Call TeamClassifier.fit(...) before predict(...).")
        color = extract_jersey_color(frame, bbox)
        if color is None:
            return None
        scaled = self._scaler.transform(color.reshape(1, -1))
        return int(self._kmeans.predict(scaled)[0])

    @property
    def cluster_swatches(self) -> List[Tuple[int, int, int]]:
        """RGB colour swatches for each cluster centre's (hue, saturation) -
        a quick visual check of which cluster id corresponds to which team's
        kit colour. The value-std feature isn't visualisable as a colour, so
        it's dropped here (a fixed brightness is used instead); a swatch
        that looks like a washed-out grey for a team you know wears a
        strongly patterned kit is expected and not a bug."""
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("Call TeamClassifier.fit(...) before cluster_swatches.")
        centers = self._scaler.inverse_transform(self._kmeans.cluster_centers_)
        swatches = []
        for hue, sat, _value_std in centers:
            hsv_pixel = np.uint8([[[np.clip(hue, 0, 179), np.clip(sat, 0, 255), 200]]])
            bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
            swatches.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
        return swatches


def resolve_track_team_ids(
    tracks_df: pd.DataFrame,
    class_names: Iterable[str] = DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES,
) -> pd.DataFrame:
    """Replaces noisy per-frame `team_id` predictions with one stable label
    per track (majority vote across all of that track's frames) - the label
    a downstream spatial analysis should actually use, since a single frame's
    prediction can flip due to occlusion, motion blur, or a bad crop.

    The original per-frame predictions are kept as `team_id_raw` for
    inspection/debugging.
    """
    class_names = set(class_names)
    df = tracks_df.rename(columns={"team_id": "team_id_raw"})
    eligible_rows = df[df["class_name"].isin(class_names) & df["team_id_raw"].notna()]
    majority = eligible_rows.groupby("track_id")["team_id_raw"].agg(lambda s: s.mode().iloc[0])
    df["team_id"] = df["track_id"].map(majority)
    return df


def resolve_goalkeeper_team_ids(
    tracks_df: pd.DataFrame,
    goalkeeper_class_name: str = "goalkeeper",
    player_class_name: str = "player",
) -> pd.DataFrame:
    """Assigns `team_id` to goalkeeper tracks by proximity to each already
    colour-resolved team's centroid, rather than jersey colour - goalkeeper
    kits are conventionally a third, distinct colour that a 2-cluster colour
    classifier can't place correctly. No-op if `tracks_df` has no detections
    of `goalkeeper_class_name` (e.g. when using the generic COCO model).

    Must run after `resolve_track_team_ids`, since it relies on outfield
    players already having a resolved `team_id`.
    """
    df = tracks_df.copy()
    if goalkeeper_class_name not in df["class_name"].unique():
        return df

    player_rows = df[(df["class_name"] == player_class_name) & df["team_id"].notna()]
    centroids = player_rows.groupby("team_id")[["pitch_x", "pitch_y"]].mean()
    if len(centroids) < 2:
        return df  # not enough resolved teams to compare a goalkeeper against

    centroid_ids = centroids.index.to_numpy()
    centroid_xy = centroids.to_numpy()

    gk_rows = df[df["class_name"] == goalkeeper_class_name]
    gk_track_positions = gk_rows.groupby("track_id")[["pitch_x", "pitch_y"]].mean()

    gk_team_id = {}
    for track_id, (x, y) in gk_track_positions.iterrows():
        dists = np.linalg.norm(centroid_xy - [x, y], axis=1)
        gk_team_id[track_id] = centroid_ids[dists.argmin()]

    gk_mask = df["class_name"] == goalkeeper_class_name
    df.loc[gk_mask, "team_id"] = df.loc[gk_mask, "track_id"].map(gk_team_id)
    return df

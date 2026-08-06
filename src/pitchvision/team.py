from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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


def _torso_crop(
    frame: np.ndarray,
    bbox: np.ndarray,
    top: float = 0.15,
    bottom: float = 0.55,
    horizontal_margin: float = 0.25,
) -> Optional[np.ndarray]:
    """A tight central patch of a bounding box's crop, targeting the chest -
    a vertical band from `top` to `bottom` of the box height (skipping the
    head/neck near the very top and everything below the torso), trimmed by
    `horizontal_margin` on each side.

    Detector boxes usually aren't pixel-tight around the visible player, and
    a wider crop's edges can carry a real amount of background - which
    matters a lot when the pitch's grass renders as yellowish-green (some
    lighting/turf conditions do), close enough in hue to a yellow kit that a
    coarse hue-range filter can't reliably tell "yellow kit" from "yellowish
    grass" apart. A tight central crop reduces reliance on that filter by
    physically containing far less background to begin with. Returns None if
    the box is degenerate (e.g. clipped fully outside the frame)."""
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    x1, y1 = max(x1, 0.0), max(y1, 0.0)
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    w, h = x2 - x1, y2 - y1

    cx1, cx2 = x1 + horizontal_margin * w, x2 - horizontal_margin * w
    cy1, cy2 = y1 + top * h, y1 + bottom * h

    crop = frame[int(cy1):int(cy2), int(cx1):int(cx2)]
    if crop.size == 0:
        return None
    return crop


def extract_jersey_color(frame: np.ndarray, bbox: np.ndarray, **crop_kwargs) -> Optional[np.ndarray]:
    """A colour signature for a player's jersey, as a 3-element feature
    vector: (median hue, median saturation, standard deviation of value)
    over a tight torso crop (see `_torso_crop`), with any remaining
    pitch-grass-coloured pixels masked out as a secondary safety net.

    `**crop_kwargs` (`top`, `bottom`, `horizontal_margin`) are forwarded to
    `_torso_crop` - tune these if the defaults crop too little (still
    picking up background) or too much (barely any jersey pixels left) for a
    given clip's box tightness and camera distance.

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
    torso = _torso_crop(frame, bbox, **crop_kwargs)
    if torso is None:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)

    hue, sat = hsv[:, 0], hsv[:, 1]
    is_grass = (hue > 35) & (hue < 85) & (sat > 40)
    kept = hsv[~is_grass]
    if len(kept) < 10:
        kept = hsv  # the mask removed almost everything; fall back to the full crop

    return np.array([np.median(kept[:, 0]), np.median(kept[:, 1]), np.std(kept[:, 2])])


@dataclass
class JerseySample:
    """One jersey-colour training sample: the feature vector `extract_jersey_color`
    produced, alongside the actual torso crop it was computed from - kept
    around so `plot_jersey_color_samples` can show what the clustering is
    really looking at, not just its numeric summary."""

    color: np.ndarray
    crop: np.ndarray


def collect_jersey_samples(
    video_path: str,
    detector: PlayerBallDetector,
    class_names: Iterable[str] = DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES,
    stride: int = 30,
    max_samples: int = 500,
    **crop_kwargs,
) -> List[JerseySample]:
    """Samples every `stride`-th frame of a clip, detects players, and
    collects their jersey-colour signatures plus the crop each one came from
    - the training data for `TeamClassifier.fit` (via `collect_jersey_colors`,
    which discards the crops) and for `plot_jersey_color_samples` (which
    needs them). Sampling across the whole clip (rather than a single frame)
    covers players under different lighting/poses/occlusion than any one
    frame would.

    `detector` is used with whichever classes it was constructed with (or
    overridden via `detector.detect(classes=...)` elsewhere); only detections
    whose *class name* is in `class_names` are kept. Using the specialized
    football-player-detection.pt checkpoint (class name "player") instead of
    the generic COCO "person" class keeps referees out of the training data,
    giving a cleaner 2-cluster fit. `**crop_kwargs` are forwarded to
    `extract_jersey_color`/`_torso_crop` - see their docstrings.
    """
    class_names = set(class_names)
    frames = VideoFrames(video_path)
    samples: List[JerseySample] = []
    try:
        for i, frame in enumerate(frames):
            if i % stride != 0:
                continue
            for det in detector.detect(frame):
                if det.class_name not in class_names:
                    continue
                color = extract_jersey_color(frame, det.xyxy, **crop_kwargs)
                crop = _torso_crop(frame, det.xyxy, **crop_kwargs)
                if color is not None and crop is not None:
                    samples.append(JerseySample(color=color, crop=crop))
            if len(samples) >= max_samples:
                break
    finally:
        frames.release()
    return samples[:max_samples]


def collect_jersey_colors(
    video_path: str,
    detector: PlayerBallDetector,
    class_names: Iterable[str] = DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES,
    stride: int = 30,
    max_samples: int = 500,
    **crop_kwargs,
) -> np.ndarray:
    """Like `collect_jersey_samples`, but returns just the colour feature
    vectors - the training data for `TeamClassifier.fit`."""
    samples = collect_jersey_samples(video_path, detector, class_names, stride, max_samples, **crop_kwargs)
    if not samples:
        return np.empty((0, 3))
    return np.array([s.color for s in samples])


def plot_jersey_color_samples(
    samples: List[JerseySample],
    cluster_labels: Iterable[Optional[int]],
    cluster_colors: Optional[Dict[int, str]] = None,
    max_thumbnails: int = 40,
    n_cols: int = 10,
) -> None:
    """Visual audit of a fitted `TeamClassifier`: a scatter of (hue,
    saturation) coloured by cluster assignment (marker size ~ pattern
    contrast, the value-std feature), and a grid of the actual torso crops
    used, bordered by their assigned cluster colour.

    Use this *before* trusting a fit - it's the fastest way to tell whether
    the clustering is actually splitting on jersey colour, or on something
    else entirely (grass bleed-through into a loose crop, shadow/lighting,
    motion blur), which raw cluster sizes or a pitch-position plot alone
    can't distinguish.
    """
    import matplotlib.pyplot as plt

    cluster_labels = np.array(list(cluster_labels), dtype=object)
    cluster_colors = cluster_colors or {}
    colors_arr = np.array([s.color for s in samples])

    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in sorted({c for c in cluster_labels if c is not None}):
        mask = cluster_labels == cluster_id
        ax.scatter(
            colors_arr[mask, 0], colors_arr[mask, 1],
            s=20 + colors_arr[mask, 2], label=f"cluster {cluster_id}",
            color=cluster_colors.get(cluster_id),
        )
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_title("Sampled jersey colours (marker size ~ pattern contrast)")
    ax.legend()
    plt.show()

    n = min(len(samples), max_thumbnails)
    n_cols = min(n_cols, max(n, 1))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.4 * n_rows))
    axes = np.atleast_2d(axes)
    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n:
            ax.axis("off")
            continue
        sample, cluster_id = samples[idx], cluster_labels[idx]
        ax.imshow(cv2.cvtColor(sample.crop, cv2.COLOR_BGR2RGB))
        color = cluster_colors.get(cluster_id, "black")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    fig.suptitle("Sample crops, bordered by assigned cluster")
    plt.tight_layout()
    plt.show()


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

    def predict_from_colors(self, colors: np.ndarray) -> np.ndarray:
        """Predicts cluster ids for already-extracted colour feature vectors
        (e.g. from `collect_jersey_samples`), without re-extracting from a
        frame/bbox like `predict` does - mainly useful for diagnostics, such
        as feeding `plot_jersey_color_samples` the cluster label for every
        sample used to fit this classifier."""
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("Call TeamClassifier.fit(...) before predict_from_colors(...).")
        colors = np.asarray(colors, dtype=np.float64)
        scaled = self._scaler.transform(colors)
        return self._kmeans.predict(scaled)

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

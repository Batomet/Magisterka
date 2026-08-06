from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.optimize import linear_sum_assignment

DEFAULT_OUTFIELD_CLASS_NAMES = ("player", "person")


def compute_track_speeds(tracks_df: pd.DataFrame, fps: float, smoothing_window: int = 3) -> pd.DataFrame:
    """Adds a `speed_mps` column to a copy of `tracks_df`: each track's
    frame-to-frame displacement in pitch metres, divided by the actual
    elapsed time between those frames (handles small tracking gaps
    correctly, rather than assuming every frame was seen).

    Positions are smoothed (a centred rolling mean, `smoothing_window`
    frames) before differencing, purely for the speed calculation - a
    stationary player's box still jitters a little frame to frame, and once
    that pixel jitter is pushed through the homography it can look like a
    meaningful few m/s of "speed" even when nobody moved, especially far
    from the camera where the pixel-to-metre scale is coarser. Unsmoothed
    `pitch_x`/`pitch_y` in the returned frame are untouched - only the speed
    values are computed from the smoothed positions.

    Each track's first observed frame has no prior frame to diff against
    and gets `speed_mps = NaN`.
    """
    df = tracks_df.sort_values(["track_id", "frame"]).copy()
    df["_smooth_x"] = df.groupby("track_id")["pitch_x"].transform(
        lambda s: s.rolling(smoothing_window, center=True, min_periods=1).mean()
    )
    df["_smooth_y"] = df.groupby("track_id")["pitch_y"].transform(
        lambda s: s.rolling(smoothing_window, center=True, min_periods=1).mean()
    )
    dt = df.groupby("track_id")["frame"].diff() / fps
    dx = df.groupby("track_id")["_smooth_x"].diff()
    dy = df.groupby("track_id")["_smooth_y"].diff()
    df["speed_mps"] = np.hypot(dx, dy) / dt
    return df.drop(columns=["_smooth_x", "_smooth_y"])


def _first_sustained_true_index(mask: Sequence[bool], min_run: int) -> Optional[int]:
    """Position (not label) of the first element starting a run of >=
    `min_run` consecutive truthy values in `mask`. None if no such run."""
    run = 0
    for i, val in enumerate(mask):
        run = run + 1 if val else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def detect_restart_frame(
    tracks_df: pd.DataFrame,
    fps: float,
    ball_class_name: str = "ball",
    speed_threshold_mps: float = 2.0,
    min_consecutive_frames: int = 3,
) -> Optional[int]:
    """Detects the restart of play after a set piece / break: the first
    frame at which the ball's speed exceeds `speed_threshold_mps`, sustained
    for `min_consecutive_frames` consecutive tracked frames - i.e. the ball
    being put back into play, rather than a single noisy detection.

    Returns None if the ball was never confidently tracked in this clip, or
    no such sustained speed increase was found (both common - ball detection
    is the least reliable part of the pipeline; see `PlayerBallDetector`'s
    docstring). Requires the specialized player-detection model's `"ball"`
    class (`SPORTS_DETECTION_CLASSES`) - the generic COCO model's `"sports
    ball"` class name differs and won't match by default.
    """
    ball_rows = tracks_df[tracks_df["class_name"] == ball_class_name]
    if ball_rows.empty:
        return None
    ball_rows = compute_track_speeds(ball_rows, fps).sort_values("frame").reset_index(drop=True)
    moving = (ball_rows["speed_mps"].fillna(0) >= speed_threshold_mps).to_numpy()
    idx = _first_sustained_true_index(moving, min_consecutive_frames)
    if idx is None:
        return None
    return int(ball_rows.loc[idx, "frame"])


def detect_team_dynamic_frame(
    tracks_df: pd.DataFrame,
    team_id: int,
    fps: float,
    class_names: Iterable[str] = DEFAULT_OUTFIELD_CLASS_NAMES,
    speed_threshold_mps: float = 1.5,
    min_fraction_moving: float = 0.5,
    min_consecutive_frames: int = 3,
    after_frame: Optional[int] = None,
) -> Optional[int]:
    """Detects the frame at which a team's formation turns dynamic: the
    first frame (after `after_frame`, if given - typically the restart
    frame from `detect_restart_frame`) at which at least
    `min_fraction_moving` of that team's tracked outfield players have speed
    >= `speed_threshold_mps`, sustained for `min_consecutive_frames`
    consecutive frames. Returns None if no such frame is found.
    """
    class_names = set(class_names)
    team_rows = tracks_df[
        (tracks_df["team_id"] == team_id) & tracks_df["class_name"].isin(class_names)
    ]
    if after_frame is not None:
        team_rows = team_rows[team_rows["frame"] > after_frame]
    if team_rows.empty:
        return None

    team_rows = compute_track_speeds(team_rows, fps)
    fraction_moving = (
        team_rows.assign(_moving=team_rows["speed_mps"].fillna(0) >= speed_threshold_mps)
        .groupby("frame")["_moving"]
        .mean()
        .sort_index()
    )
    idx = _first_sustained_true_index((fraction_moving >= min_fraction_moving).to_numpy(), min_consecutive_frames)
    if idx is None:
        return None
    return int(fraction_moving.index[idx])


def compute_transition_time(
    tracks_df: pd.DataFrame,
    team_id: int,
    fps: float,
    class_names: Iterable[str] = DEFAULT_OUTFIELD_CLASS_NAMES,
    ball_class_name: str = "ball",
    ball_speed_threshold_mps: float = 2.0,
    team_speed_threshold_mps: float = 1.5,
    min_fraction_moving: float = 0.5,
    min_consecutive_frames: int = 3,
) -> Optional[Dict[str, float]]:
    """Measures how long, after the restart of play, a team took to
    transition from a static to a dynamic formation - the restart is
    detected from the ball's movement onset (`detect_restart_frame`), the
    transition from a sustained fraction of the team's players starting to
    move (`detect_team_dynamic_frame`).

    Returns None if either couldn't be detected (most commonly: the ball was
    never confidently tracked in this clip). Otherwise a dict with
    `restart_frame`, `dynamic_frame`, `transition_frames`, and
    `transition_seconds`.
    """
    restart_frame = detect_restart_frame(
        tracks_df, fps, ball_class_name, ball_speed_threshold_mps, min_consecutive_frames
    )
    if restart_frame is None:
        return None
    dynamic_frame = detect_team_dynamic_frame(
        tracks_df, team_id, fps, class_names, team_speed_threshold_mps,
        min_fraction_moving, min_consecutive_frames, after_frame=restart_frame,
    )
    if dynamic_frame is None:
        return None
    transition_frames = dynamic_frame - restart_frame
    return {
        "restart_frame": restart_frame,
        "dynamic_frame": dynamic_frame,
        "transition_frames": transition_frames,
        "transition_seconds": transition_frames / fps,
    }


def extract_static_formation(
    tracks_df: pd.DataFrame,
    team_id: int,
    restart_frame: int,
    lookback_frames: int = 15,
    class_names: Iterable[str] = DEFAULT_OUTFIELD_CLASS_NAMES,
) -> np.ndarray:
    """A team's average player positions over the `lookback_frames` frames
    immediately before `restart_frame` - the "static formation" snapshot for
    one set-piece instance, the input `compute_formation_repeatability`
    compares across instances. Averaging over a short window (rather than a
    single frame) smooths out per-frame tracking jitter for players who are
    genuinely standing still waiting for the restart.

    Positions are averaged per track_id, so the number of rows returned is
    the number of distinct tracks seen for this team in that window - not
    necessarily the same across instances if a player wasn't tracked in
    every frame of the window. Check `len(...)` across instances before
    trusting a repeatability comparison between them.
    """
    class_names = set(class_names)
    window = tracks_df[
        (tracks_df["team_id"] == team_id)
        & tracks_df["class_name"].isin(class_names)
        & (tracks_df["frame"] < restart_frame)
        & (tracks_df["frame"] >= restart_frame - lookback_frames)
    ]
    return window.groupby("track_id")[["pitch_x", "pitch_y"]].mean().to_numpy()


def match_formations(positions_a: np.ndarray, positions_b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Optimal one-to-one matching between two player-position sets (e.g.
    the same team's static formation at two different corners), via the
    Hungarian algorithm minimising total assignment distance.

    A player's track_id doesn't carry over between separate set-piece clips
    (no persistent player re-identification here), so this is how two
    formations are compared without needing player identity: it finds the
    pairing between the two position sets that makes them look as similar
    as possible, then reports how similar that best-case pairing actually
    is - a meaningful "these two formations match" score even though we
    don't know which specific player is which.

    positions_a/positions_b: (n, 2) pitch coordinates. If they don't have
    the same length, only `min(len(a), len(b))` pairs are matched; leftover
    points in the larger set are ignored.

    Returns (col_indices, mean_distance_m): `col_indices[i]` is the index
    into `positions_b` matched to `positions_a[i]` for `i` in the matched
    row indices (see `scipy.optimize.linear_sum_assignment`); `mean_distance_m`
    is the average distance (metres) over the matched pairs - the
    repeatability score for this pair of instances (lower = more repeatable).
    """
    positions_a = np.asarray(positions_a, dtype=np.float64)
    positions_b = np.asarray(positions_b, dtype=np.float64)
    cost = np.linalg.norm(positions_a[:, None, :] - positions_b[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    mean_distance = float(cost[row_ind, col_ind].mean())
    return col_ind, mean_distance


def compute_formation_repeatability(instances: List[np.ndarray]) -> Dict[str, float]:
    """Repeatability of a team's positional structure across multiple
    instances of the same restart type (e.g. several corners taken from the
    same side) - each element of `instances` is that team's (n, 2) player
    positions at the static moment for one instance, from
    `extract_static_formation`.

    For every pair of instances, finds the best-case (Hungarian-matched)
    alignment and its mean distance (`match_formations`), then reports the
    average of those pairwise distances across all pairs - the overall
    repeatability score (lower = more consistent structure from one
    instance to the next). Needs at least 2 instances to compare.
    """
    if len(instances) < 2:
        raise ValueError("Need at least 2 set-piece instances to assess repeatability.")
    pairwise_distances = []
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            _, mean_distance = match_formations(instances[i], instances[j])
            pairwise_distances.append(mean_distance)
    pairwise_distances = np.array(pairwise_distances)
    return {
        "n_instances": len(instances),
        "n_pairs": len(pairwise_distances),
        "mean_pairwise_distance_m": float(pairwise_distances.mean()),
        "std_pairwise_distance_m": float(pairwise_distances.std()),
        "max_pairwise_distance_m": float(pairwise_distances.max()),
    }


def pick_team_by_swatch_color(
    swatches: Sequence[Tuple[int, int, int]], reference_rgb: Tuple[int, int, int]
) -> int:
    """Picks the cluster id whose `TeamClassifier.cluster_swatches` colour is
    closest to `reference_rgb`. Needed because `TeamClassifier` is fit
    independently per clip, so cluster id 0/1 isn't guaranteed to mean the
    same real team from one clip to the next - when comparing the same
    team's formation across several separate set-piece clips, use this
    (with e.g. the analysed team's known kit colour) to consistently resolve
    "which cluster id is that team" in each clip rather than assuming id 0
    always means the same team.
    """
    distances = [sum((a - b) ** 2 for a, b in zip(swatch, reference_rgb)) for swatch in swatches]
    return int(np.argmin(distances))


def plot_formations(
    ax: Axes,
    instances: List[np.ndarray],
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
) -> None:
    """Overlays multiple formation instances (e.g. several corners) onto a
    pitch axes (typically `pitchvision.pitch.draw_pitch()`), each in a
    different colour, to visually inspect repeatability alongside the
    numeric `compute_formation_repeatability` score."""
    default_colors = plt.get_cmap("tab10").colors
    for i, positions in enumerate(instances):
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        label = labels[i] if labels else f"instance {i}"
        ax.scatter(positions[:, 0], positions[:, 1], color=color, label=label, edgecolors="black", s=60, zorder=3)
    ax.legend()


def plot_team_speed_timeline(
    ax: Axes,
    tracks_df: pd.DataFrame,
    team_id: int,
    fps: float,
    class_names: Iterable[str] = DEFAULT_OUTFIELD_CLASS_NAMES,
    restart_frame: Optional[int] = None,
    dynamic_frame: Optional[int] = None,
) -> None:
    """Plots a team's mean player speed per frame onto `ax`, with vertical
    markers for the detected restart (`detect_restart_frame`) and dynamic
    transition (`detect_team_dynamic_frame`) frames, if given - a visual
    check on whether the automatic detection actually landed on the real
    restart/transition moments."""
    class_names = set(class_names)
    team_rows = compute_track_speeds(
        tracks_df[(tracks_df["team_id"] == team_id) & tracks_df["class_name"].isin(class_names)], fps
    )
    mean_speed = team_rows.groupby("frame")["speed_mps"].mean().sort_index()
    ax.plot(mean_speed.index, mean_speed.to_numpy(), label=f"Team {team_id} mean speed")
    if restart_frame is not None:
        ax.axvline(restart_frame, color="black", linestyle="--", label="Restart")
    if dynamic_frame is not None:
        ax.axvline(dynamic_frame, color="red", linestyle=":", label="Dynamic transition")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean speed (m/s)")
    ax.legend()

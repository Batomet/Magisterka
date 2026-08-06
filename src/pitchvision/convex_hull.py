import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPoint, Polygon

from .pitch import draw_pitch

DEFAULT_HULL_CLASS_NAMES = ("player", "person")


def convex_hull_polygon(positions: np.ndarray) -> Polygon:
    """The convex hull of a set of (x, y) pitch positions, as a shapely
    Polygon. Needs at least 3 non-collinear points to form a genuine 2D
    shape; fewer points (or collinear ones, e.g. a back line standing in a
    near-perfect row) produce a degenerate result, returned here as an empty
    Polygon (zero area) rather than raising - "no meaningful playing space
    this frame" is a valid outcome, not an error."""
    positions = np.asarray(positions, dtype=np.float64)
    hull = MultiPoint([tuple(p) for p in positions]).convex_hull
    if hull.geom_type != "Polygon":
        return Polygon()
    return hull


def compute_frame_convex_hull(positions: np.ndarray) -> Optional[Dict[str, float]]:
    """Convex-hull metrics for a set of player positions in a single frame -
    the "Effective Playing Space" a team (or both teams combined) is using
    at that moment. `positions`: (n, 2) pitch (x, y) coordinates in metres.
    Returns None if fewer than 3 players (a hull needs at least a triangle).
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 3:
        return None
    hull = convex_hull_polygon(positions)
    return {
        "n_players": len(positions),
        "area_m2": hull.area,
        "perimeter_m": hull.length,
    }


def compute_team_convex_hull(
    tracks_df: pd.DataFrame,
    team_id: int,
    class_names: Iterable[str] = DEFAULT_HULL_CLASS_NAMES,
) -> pd.DataFrame:
    """Per-frame convex-hull ("Effective Playing Space") metrics for one
    team across a tracked clip - the area/perimeter of the smallest polygon
    containing all of that team's tracked outfield players.

    Goalkeepers excluded by default, matching
    `compactness.compute_team_compactness`: a goalkeeper anchored deep in
    their own box would inflate a midfield/build-up hull in a way that
    doesn't reflect the outfield shape being analysed. Expects `tracks_df`
    as produced by `pitchvision.pipeline.TrackingPipeline`.
    """
    class_names = set(class_names)
    team_rows = tracks_df[
        (tracks_df["team_id"] == team_id) & tracks_df["class_name"].isin(class_names)
    ]
    records = []
    for frame, group in team_rows.groupby("frame"):
        metrics = compute_frame_convex_hull(group[["pitch_x", "pitch_y"]].to_numpy())
        if metrics is None:
            continue
        metrics["frame"] = frame
        metrics["team_id"] = team_id
        records.append(metrics)
    columns = ["frame", "team_id", "n_players", "area_m2", "perimeter_m"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]


def compute_combined_convex_hull(
    tracks_df: pd.DataFrame,
    class_names: Iterable[str] = DEFAULT_HULL_CLASS_NAMES,
) -> pd.DataFrame:
    """Per-frame convex-hull metrics over every on-pitch outfield player from
    both teams combined - the classical "Effective Playing Space" definition
    (Frencken et al., 2011): the surface area of the smallest polygon
    containing all outfield players, i.e. the total pitch area actively in
    use by play at that moment. Complements
    `compute_team_convex_hull`'s per-team breakdown - the combined figure
    can shrink even while one team's own hull grows, if the two teams'
    players are moving into overlapping space rather than spreading apart.
    """
    class_names = set(class_names)
    rows_df = tracks_df[tracks_df["class_name"].isin(class_names)]
    records = []
    for frame, group in rows_df.groupby("frame"):
        metrics = compute_frame_convex_hull(group[["pitch_x", "pitch_y"]].to_numpy())
        if metrics is None:
            continue
        metrics["frame"] = frame
        records.append(metrics)
    columns = ["frame", "n_players", "area_m2", "perimeter_m"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]


def plot_convex_hull(ax: Axes, positions: np.ndarray, color: str = "yellow", alpha: float = 0.25) -> None:
    """Draws the convex hull of `positions` onto `ax` (typically a
    `pitchvision.pitch.draw_pitch()` axes). No-op if fewer than 3 points."""
    hull = convex_hull_polygon(positions)
    if hull.is_empty:
        return
    coords = list(hull.exterior.coords)
    ax.add_patch(
        MplPolygon(coords, closed=True, facecolor=color, edgecolor=color, linewidth=2, alpha=alpha)
    )


def save_convex_hull_frames(
    tracks_df: pd.DataFrame,
    output_dir: str,
    frames: Optional[Iterable[int]] = None,
    stride: int = 1,
    class_names: Iterable[str] = DEFAULT_HULL_CLASS_NAMES,
    team_colors: Optional[Dict[int, str]] = None,
    filename_prefix: str = "convex_hull_frame",
) -> List[str]:
    """Renders and saves a per-team convex-hull plot (pitch + each team's
    hull outline + player dots) for a set of frames, one PNG per frame, to
    `output_dir/{filename_prefix}_{frame:05d}.png`. Returns the list of
    saved file paths, in frame order.

    Works on *any* DataFrame shaped like `TrackingPipeline.run()`'s output -
    same `frame`/`class_name`/`team_id`/`pitch_x`/`pitch_y` columns as
    `voronoi.save_voronoi_frames` - so a hand-built or previously-saved CSV
    works identically to a live pipeline result.

    `frames`: which frame numbers to render. If not given, defaults to every
    frame with >=3 players on at least one team (i.e. every frame with a
    meaningful hull to draw for at least one team), subsampled by `stride`.
    """
    class_names = set(class_names)
    eligible = tracks_df[
        tracks_df["class_name"].isin(class_names) & tracks_df["team_id"].notna()
    ]

    if frames is None:
        candidate_frames = sorted(
            frame
            for frame, group in eligible.groupby("frame")
            if (group.groupby("team_id").size() >= 3).any()
        )
        frames = candidate_frames[::stride]

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for frame in frames:
        frame_rows = eligible[eligible["frame"] == frame]
        if frame_rows.empty:
            continue

        ax = draw_pitch()
        for team_id, group in frame_rows.groupby("team_id"):
            color = (team_colors or {}).get(team_id, "gray")
            positions = group[["pitch_x", "pitch_y"]].to_numpy()
            if len(positions) >= 3:
                plot_convex_hull(ax, positions, color=color)
            ax.scatter(
                positions[:, 0], positions[:, 1], color=color,
                edgecolors="black", s=60, zorder=3,
            )
        ax.set_title(f"Effective Playing Space at frame {int(frame)}")

        path = os.path.join(output_dir, f"{filename_prefix}_{int(frame):05d}.png")
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        saved_paths.append(path)

    return saved_paths

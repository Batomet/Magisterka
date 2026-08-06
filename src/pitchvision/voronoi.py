import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

from .config import PITCH_LENGTH_M, PITCH_WIDTH_M
from .pitch import draw_pitch


def pitch_voronoi_cells(
    points: np.ndarray,
    length: float = PITCH_LENGTH_M,
    width: float = PITCH_WIDTH_M,
) -> List[Polygon]:
    """Voronoi tessellation of the pitch by `points` (n, 2) pitch coordinates,
    one polygon per point, each clipped to the pitch rectangle
    [0, length] x [0, width].

    This is the standard simplified "space control" model used in tactical
    analysis: the region of the pitch closer to a given player than to any
    other (by straight-line distance) is credited to that player. It ignores
    player speed, orientation, and reaction time - unlike more advanced
    "pitch control" models (e.g. Spearman et al.) - but is a well-established
    first-order approximation and computationally trivial.

    `scipy.spatial.Voronoi` only produces *bounded* regions for points whose
    region doesn't touch the outer convex hull of the input; for players
    spread across a pitch, most of them are on that hull and would get
    unbounded regions extending to infinity. Rather than reconstructing those
    manually, we add a ring of dummy points far outside the pitch, which
    "caps" every real point's region into something finite, then clip every
    resulting polygon down to the actual pitch rectangle - the only boundary
    that matters for interpreting the areas as controlled pitch space.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points for a Voronoi diagram.")

    pad = 10 * max(length, width)
    cx, cy = length / 2, width / 2
    far_points = np.array(
        [
            (cx - pad, cy - pad), (cx + pad, cy - pad),
            (cx - pad, cy + pad), (cx + pad, cy + pad),
            (cx, cy - pad), (cx, cy + pad), (cx - pad, cy), (cx + pad, cy),
        ]
    )
    vor = Voronoi(np.vstack([points, far_points]))

    pitch_box = box(0, 0, length, width)
    polygons = []
    for i in range(n):
        region_index = vor.point_region[i]
        vertex_indices = vor.regions[region_index]
        if not vertex_indices or -1 in vertex_indices:
            # Shouldn't happen given the far-point padding above, but fall
            # back to an empty polygon rather than crash on a degenerate input.
            polygons.append(Polygon())
            continue
        polygon = Polygon(vor.vertices[vertex_indices])
        polygons.append(polygon.intersection(pitch_box))
    return polygons


def voronoi_cell_areas(points: np.ndarray, **kwargs) -> np.ndarray:
    """Area (m^2) of each point's Voronoi cell, clipped to the pitch."""
    return np.array([p.area for p in pitch_voronoi_cells(points, **kwargs)])


def team_space_control(
    positions: np.ndarray, team_ids: np.ndarray, **kwargs
) -> Dict[int, float]:
    """Total Voronoi-cell area (m^2) controlled by each team for one frame,
    given all players' pitch positions and their team_id. Uses every
    player's position (both teams) to build the tessellation, since a
    player's controlled space is bounded by opponents too, not just
    teammates."""
    areas = voronoi_cell_areas(positions, **kwargs)
    team_ids = np.asarray(team_ids)
    return {int(t): float(areas[team_ids == t].sum()) for t in np.unique(team_ids)}


def compute_space_control(
    tracks_df: pd.DataFrame,
    class_names: Iterable[str] = ("player", "goalkeeper", "person"),
) -> pd.DataFrame:
    """Per-frame total pitch area (m^2) controlled by each team, via Voronoi
    tessellation over all on-pitch players. Goalkeepers are included by
    default (unlike `compactness.compute_team_compactness`) since they still
    occupy and influence space near their own goal, which matters for space
    control even though it's excluded from outfield defensive-shape metrics.

    Expects `tracks_df` as produced by `pitchvision.pipeline.TrackingPipeline`.
    Frames with fewer than 2 players, or with only one resolved team, are
    skipped (no tessellation possible / no team comparison to make).
    """
    class_names = set(class_names)
    eligible = tracks_df[
        tracks_df["class_name"].isin(class_names) & tracks_df["team_id"].notna()
    ]
    rows = []
    for frame, group in eligible.groupby("frame"):
        if len(group) < 2 or group["team_id"].nunique() < 2:
            continue
        positions = group[["pitch_x", "pitch_y"]].to_numpy()
        team_ids = group["team_id"].to_numpy()
        totals = team_space_control(positions, team_ids)
        row = {"frame": frame}
        row.update({f"team_{team_id}_area_m2": area for team_id, area in totals.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_voronoi(
    ax: Axes,
    polygons: List[Polygon],
    team_ids: Iterable[Optional[int]],
    team_colors: Optional[Dict[int, str]] = None,
    alpha: float = 0.35,
) -> None:
    """Draws Voronoi cells (as returned by `pitch_voronoi_cells`) onto `ax`
    (typically a `pitchvision.pitch.draw_pitch()` axes), coloured by team."""
    team_colors = team_colors or {}
    for polygon, team_id in zip(polygons, team_ids):
        if polygon.is_empty:
            continue
        color = team_colors.get(team_id, "gray")
        coords = list(polygon.exterior.coords)
        ax.add_patch(
            MplPolygon(coords, closed=True, facecolor=color, edgecolor="white", linewidth=0.5, alpha=alpha)
        )


def save_voronoi_frames(
    tracks_df: pd.DataFrame,
    output_dir: str,
    frames: Optional[Iterable[int]] = None,
    stride: int = 1,
    class_names: Iterable[str] = ("player", "goalkeeper", "person"),
    team_colors: Optional[Dict[int, str]] = None,
    filename_prefix: str = "voronoi_frame",
) -> List[str]:
    """Renders and saves a Voronoi space-control plot (pitch + coloured
    cells + player dots) for a set of frames, one PNG per frame, to
    `output_dir/{filename_prefix}_{frame:05d}.png`. Returns the list of
    saved file paths, in frame order.

    Works on *any* DataFrame shaped like `TrackingPipeline.run()`'s output -
    it only needs `frame`, `class_name`, `team_id`, `pitch_x`, `pitch_y`
    columns - so a CSV you built by hand, or loaded back from a previous
    run via `pd.read_csv(...)`, works exactly the same as a live pipeline
    result. No need to re-run detection/tracking just to regenerate plots.

    `frames`: which frame numbers to render. If not given, defaults to every
    frame in `tracks_df` that has >=2 players and >=2 resolved teams (i.e.
    every frame `compute_space_control` would also produce a row for),
    subsampled by `stride` (every `stride`-th of those, in order) - a full
    clip can have hundreds of frames, so `stride` keeps this to a manageable
    number of images by default (stride=1 renders every eligible frame).
    """
    class_names = set(class_names)
    eligible = tracks_df[
        tracks_df["class_name"].isin(class_names) & tracks_df["team_id"].notna()
    ]

    if frames is None:
        candidate_frames = sorted(
            frame
            for frame, group in eligible.groupby("frame")
            if len(group) >= 2 and group["team_id"].nunique() >= 2
        )
        frames = candidate_frames[::stride]

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for frame in frames:
        frame_rows = eligible[eligible["frame"] == frame]
        if len(frame_rows) < 2 or frame_rows["team_id"].nunique() < 2:
            continue

        positions = frame_rows[["pitch_x", "pitch_y"]].to_numpy()
        team_ids = frame_rows["team_id"].to_numpy()
        polygons = pitch_voronoi_cells(positions)

        ax = draw_pitch()
        plot_voronoi(ax, polygons, team_ids, team_colors=team_colors)
        for (x, y), team_id in zip(positions, team_ids):
            ax.scatter(
                x, y, color=(team_colors or {}).get(team_id, "gray"),
                edgecolors="black", s=60, zorder=3,
            )
        ax.set_title(f"Space control at frame {int(frame)}")

        path = os.path.join(output_dir, f"{filename_prefix}_{int(frame):05d}.png")
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        saved_paths.append(path)

    return saved_paths

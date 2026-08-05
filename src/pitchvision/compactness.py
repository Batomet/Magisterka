from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """All pairwise Euclidean distances between rows of `positions` (n, 2), as a flat array."""
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(positions), k=1)
    return dists[iu]


def compute_frame_compactness(positions: np.ndarray) -> Optional[Dict[str, float]]:
    """Compactness metrics for one team's player positions in a single frame.

    `positions`: (n, 2) array of pitch (x, y) coordinates in metres. Returns
    None if fewer than 2 players (pairwise distance undefined).

    - `mean_pairwise_distance_m`: average distance between every pair of
      players - the direct "inter-player distance" compactness measure.
    - `stretch_index_m`: average distance from the team centroid - a
      complementary compactness measure less sensitive to a single
      isolated outlier player than the pairwise mean.
    - `length_m`/`width_m`: the team's bounding-box extent along the pitch's
      length/width axes - how far the formation is stretched in each
      direction.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return None
    centroid = positions.mean(axis=0)
    pairwise = _pairwise_distances(positions)
    stretch = np.linalg.norm(positions - centroid, axis=1).mean()
    length = positions[:, 0].max() - positions[:, 0].min()
    width = positions[:, 1].max() - positions[:, 1].min()
    return {
        "n_players": len(positions),
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "mean_pairwise_distance_m": pairwise.mean(),
        "stretch_index_m": stretch,
        "length_m": length,
        "width_m": width,
    }


def compute_team_compactness(
    tracks_df: pd.DataFrame,
    team_id: int,
    class_names: Iterable[str] = ("player", "person"),
) -> pd.DataFrame:
    """Per-frame compactness metrics for one team across a tracked clip -
    the time series a "defensive compactness over the phase" analysis is
    built from. Expects `tracks_df` as produced by
    `pitchvision.pipeline.TrackingPipeline` (needs `frame`, `team_id`,
    `class_name`, `pitch_x`, `pitch_y` columns).

    Goalkeepers are excluded by default (`class_names` doesn't include
    "goalkeeper") since they occupy a structurally different role near their
    own goal and would distort an outfield defensive-line compactness
    metric; pass `class_names=("player", "goalkeeper")` to include them.
    """
    class_names = set(class_names)
    team_rows = tracks_df[
        (tracks_df["team_id"] == team_id) & tracks_df["class_name"].isin(class_names)
    ]
    records = []
    for frame, group in team_rows.groupby("frame"):
        metrics = compute_frame_compactness(group[["pitch_x", "pitch_y"]].to_numpy())
        if metrics is None:
            continue
        metrics["frame"] = frame
        metrics["team_id"] = team_id
        records.append(metrics)
    columns = [
        "frame", "team_id", "n_players", "centroid_x", "centroid_y",
        "mean_pairwise_distance_m", "stretch_index_m", "length_m", "width_m",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]

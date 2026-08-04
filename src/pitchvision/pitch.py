from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .config import (
    CENTRE_CIRCLE_RADIUS_M,
    GOAL_BOX_LENGTH_M,
    GOAL_BOX_WIDTH_M,
    PENALTY_BOX_LENGTH_M,
    PENALTY_BOX_WIDTH_M,
    PENALTY_SPOT_DISTANCE_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)


def draw_pitch(ax: Optional[Axes] = None, line_color: str = "white", pitch_color: str = "#0b6623") -> Axes:
    """Draws a top-down FIFA-regulation pitch (105 x 68 m) using the same
    top-left-origin coordinate convention as `pitchvision.calibration`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))

    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    ax.set_facecolor(pitch_color)
    ax.add_patch(Rectangle((0, 0), L, W, fill=False, edgecolor=line_color, linewidth=1.5))
    ax.plot([L / 2, L / 2], [0, W], color=line_color, linewidth=1.5)
    ax.add_patch(plt.Circle((L / 2, W / 2), CENTRE_CIRCLE_RADIUS_M, fill=False, edgecolor=line_color, linewidth=1.5))
    ax.plot(L / 2, W / 2, marker="o", color=line_color, markersize=2)

    for x0, direction in ((0, 1), (L, -1)):
        ax.add_patch(
            Rectangle(
                (x0, (W - PENALTY_BOX_WIDTH_M) / 2), direction * PENALTY_BOX_LENGTH_M, PENALTY_BOX_WIDTH_M,
                fill=False, edgecolor=line_color, linewidth=1.5,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0, (W - GOAL_BOX_WIDTH_M) / 2), direction * GOAL_BOX_LENGTH_M, GOAL_BOX_WIDTH_M,
                fill=False, edgecolor=line_color, linewidth=1.5,
            )
        )
        penalty_spot_x = PENALTY_SPOT_DISTANCE_M if direction == 1 else L - PENALTY_SPOT_DISTANCE_M
        ax.plot(penalty_spot_x, W / 2, marker="o", color=line_color, markersize=2)

    ax.set_xlim(-5, L + 5)
    ax.set_ylim(-5, W + 5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_positions(ax: Axes, positions: np.ndarray, **scatter_kwargs) -> None:
    positions = np.atleast_2d(np.asarray(positions))
    ax.scatter(positions[:, 0], positions[:, 1], **scatter_kwargs)

from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO

from ._weights import download_from_gdrive
from .calibration import PitchCalibrator
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

# Google Drive file id for the pretrained `football-pitch-detection.pt`
# weights (a YOLOv8-pose model detecting the 32 pitch keypoints below),
# published by the Roboflow "sports" project (https://github.com/roboflow/sports)
# under examples/soccer/setup.sh. Public file, no account/API key needed.
PITCH_KEYPOINT_WEIGHTS_GDRIVE_ID = "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"


def download_pitch_keypoint_weights(destination: str) -> str:
    """Downloads the pretrained pitch-keypoint weights to `destination` -
    point it at a path on Drive to avoid re-downloading every Colab
    session."""
    return download_from_gdrive(PITCH_KEYPOINT_WEIGHTS_GDRIVE_ID, destination)


def pitch_keypoint_template(
    length: float = PITCH_LENGTH_M,
    width: float = PITCH_WIDTH_M,
) -> np.ndarray:
    """The 32 standard pitch keypoints, in metres, in the top-left-origin
    convention used throughout `pitchvision.calibration`.

    Row i (0-based) is the metric position of keypoint (i + 1) as output by
    the pretrained pitch-keypoint model - this reuses the exact vertex
    ordering/topology of the `SoccerPitchConfiguration` template from
    https://github.com/roboflow/sports (sports/configs/soccer.py), since
    that's what the model was trained to detect, but computes the actual
    coordinates from real Laws-of-the-Game measurements rather than that
    project's own template scale (which is internally inconsistent with the
    Laws - e.g. it uses a 20.15 m penalty-box depth, when the Laws fix that
    at 16.5 m regardless of overall pitch size).
    """
    L, W = length, width
    pb_l, pb_w = PENALTY_BOX_LENGTH_M, PENALTY_BOX_WIDTH_M
    gb_l, gb_w = GOAL_BOX_LENGTH_M, GOAL_BOX_WIDTH_M
    r = CENTRE_CIRCLE_RADIUS_M
    spot = PENALTY_SPOT_DISTANCE_M

    return np.array(
        [
            (0, 0),                                    # 1
            (0, (W - pb_w) / 2),                        # 2
            (0, (W - gb_w) / 2),                        # 3
            (0, (W + gb_w) / 2),                        # 4
            (0, (W + pb_w) / 2),                        # 5
            (0, W),                                     # 6
            (gb_l, (W - gb_w) / 2),                      # 7
            (gb_l, (W + gb_w) / 2),                      # 8
            (spot, W / 2),                               # 9
            (pb_l, (W - pb_w) / 2),                      # 10
            (pb_l, (W - gb_w) / 2),                      # 11
            (pb_l, (W + gb_w) / 2),                      # 12
            (pb_l, (W + pb_w) / 2),                      # 13
            (L / 2, 0),                                  # 14
            (L / 2, W / 2 - r),                          # 15
            (L / 2, W / 2 + r),                          # 16
            (L / 2, W),                                  # 17
            (L - pb_l, (W - pb_w) / 2),                  # 18
            (L - pb_l, (W - gb_w) / 2),                  # 19
            (L - pb_l, (W + gb_w) / 2),                  # 20
            (L - pb_l, (W + pb_w) / 2),                  # 21
            (L - spot, W / 2),                           # 22
            (L - gb_l, (W - gb_w) / 2),                  # 23
            (L - gb_l, (W + gb_w) / 2),                  # 24
            (L, 0),                                      # 25
            (L, (W - pb_w) / 2),                         # 26
            (L, (W - gb_w) / 2),                         # 27
            (L, (W + gb_w) / 2),                         # 28
            (L, (W + pb_w) / 2),                         # 29
            (L, W),                                      # 30
            (L / 2 - r, W / 2),                          # 31
            (L / 2 + r, W / 2),                          # 32
        ],
        dtype=np.float64,
    )


class PitchKeypointDetector:
    """Detects the 32 standard pitch keypoints in a frame with a pretrained
    YOLOv8-pose model, and turns the confidently-detected ones directly into
    a fitted `PitchCalibrator` - fully automatic pitch calibration, no
    manual point picking required. Falls back to `PitchCalibrator.from_point_pairs`
    for clips where too few keypoints are confidently detected (e.g. heavy
    occlusion, an unusual camera angle).
    """

    def __init__(self, weights: str, confidence: float = 0.5, device: Optional[str] = None):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (pixel_xy, confidence), each of length 32, in the fixed
        vertex order matched by `pitch_keypoint_template`. Keypoints the
        model didn't confidently localise still get a row - filter by
        confidence before using them as correspondences."""
        result = self.model.predict(frame, device=self.device, verbose=False)[0]
        keypoints = result.keypoints
        if keypoints is None or keypoints.xy.shape[0] == 0:
            raise RuntimeError("No pitch detected in this frame.")

        instance = 0
        if keypoints.xy.shape[0] > 1:
            instance = int(result.boxes.conf.argmax())

        xy = keypoints.xy[instance].cpu().numpy()
        conf = keypoints.conf[instance].cpu().numpy()
        return xy, conf

    def calibrate(self, frame: np.ndarray, min_confidence: Optional[float] = None) -> PitchCalibrator:
        xy, conf = self.detect(frame)
        threshold = self.confidence if min_confidence is None else min_confidence
        mask = conf >= threshold
        if mask.sum() < 4:
            raise RuntimeError(
                f"Only {int(mask.sum())} confident pitch keypoints found (need >= 4). "
                "Try lowering `min_confidence`, or fall back to manual calibration "
                "for this clip via PitchCalibrator.from_point_pairs."
            )
        template = pitch_keypoint_template()
        return PitchCalibrator.from_point_pairs(xy[mask], template[mask])

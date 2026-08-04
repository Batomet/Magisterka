import cv2
import numpy as np

from .config import (
    CENTRE_CIRCLE_RADIUS_M,
    GOAL_BOX_LENGTH_M,
    GOAL_BOX_WIDTH_M,
    GOAL_WIDTH_M,
    PENALTY_BOX_LENGTH_M,
    PENALTY_BOX_WIDTH_M,
    PENALTY_SPOT_DISTANCE_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)


class PitchCalibrator:
    """Maps pixel coordinates from a broadcast/perspective camera view onto
    metric 2D pitch coordinates via a homography.

    Pitch coordinate system: origin at the pitch's top-left corner as seen
    from above, x axis along the length (0..PITCH_LENGTH_M), y axis along the
    width (0..PITCH_WIDTH_M).

    Fit the homography from >=4 known pixel<->pitch point correspondences
    (pitch landmarks such as corner flags, penalty box corners, or centre
    circle tangents are easy to identify by eye in a paused frame — see
    PITCH_LANDMARKS_M below). A single calibration is valid for as long as
    the camera doesn't pan/zoom/cut, which typically holds within a single
    broadcast phase-of-play clip.

    This point-correspondence approach is deliberately semi-automatic: the
    correspondences are supplied once per clip rather than detected by a
    learned pitch-keypoint model. Swapping in a fully automatic keypoint
    detector later only requires producing the same (pixel_points,
    pitch_points) arrays that `from_point_pairs` consumes.
    """

    def __init__(self, homography: np.ndarray):
        self.H = homography
        self.H_inv = np.linalg.inv(homography)

    @classmethod
    def from_point_pairs(cls, pixel_points: np.ndarray, pitch_points: np.ndarray) -> "PitchCalibrator":
        pixel_points = np.asarray(pixel_points, dtype=np.float64)
        pitch_points = np.asarray(pitch_points, dtype=np.float64)
        if len(pixel_points) < 4 or len(pixel_points) != len(pitch_points):
            raise ValueError(
                "At least 4 matching point correspondences are required to fit a homography."
            )
        H, _ = cv2.findHomography(pixel_points, pitch_points, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("Homography estimation failed; check the point correspondences.")
        return cls(H)

    def pixel_to_pitch(self, points: np.ndarray) -> np.ndarray:
        return self._apply(self.H, points)

    def pitch_to_pixel(self, points: np.ndarray) -> np.ndarray:
        return self._apply(self.H_inv, points)

    @staticmethod
    def _apply(H: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = homogeneous @ H.T
        transformed /= transformed[:, [2]]
        return transformed[:, :2]

    def reprojection_error(self, pixel_points: np.ndarray, pitch_points: np.ndarray) -> float:
        """Mean Euclidean error (in metres) between the given pitch points and
        the pitch points predicted from their pixel correspondences. Useful as
        a sanity check on calibration quality."""
        predicted = self.pixel_to_pitch(pixel_points)
        return float(np.mean(np.linalg.norm(predicted - np.asarray(pitch_points), axis=1)))


# Common pitch landmarks in metres, using the top-left-origin convention above.
# Only use the subset that is actually visible in a given frame when calibrating.
PITCH_LANDMARKS_M = {
    "top_left_corner": (0.0, 0.0),
    "bottom_left_corner": (0.0, PITCH_WIDTH_M),
    "top_right_corner": (PITCH_LENGTH_M, 0.0),
    "bottom_right_corner": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "centre_top": (PITCH_LENGTH_M / 2, 0.0),
    "centre_bottom": (PITCH_LENGTH_M / 2, PITCH_WIDTH_M),
    "centre_spot": (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2),
    # Penalty box.
    "left_penalty_top": (PENALTY_BOX_LENGTH_M, (PITCH_WIDTH_M - PENALTY_BOX_WIDTH_M) / 2),
    "left_penalty_bottom": (PENALTY_BOX_LENGTH_M, (PITCH_WIDTH_M + PENALTY_BOX_WIDTH_M) / 2),
    "right_penalty_top": (PITCH_LENGTH_M - PENALTY_BOX_LENGTH_M, (PITCH_WIDTH_M - PENALTY_BOX_WIDTH_M) / 2),
    "right_penalty_bottom": (PITCH_LENGTH_M - PENALTY_BOX_LENGTH_M, (PITCH_WIDTH_M + PENALTY_BOX_WIDTH_M) / 2),
    "left_penalty_spot": (PENALTY_SPOT_DISTANCE_M, PITCH_WIDTH_M / 2),
    "right_penalty_spot": (PITCH_LENGTH_M - PENALTY_SPOT_DISTANCE_M, PITCH_WIDTH_M / 2),
    # Six-yard box.
    "left_six_yard_top": (GOAL_BOX_LENGTH_M, (PITCH_WIDTH_M - GOAL_BOX_WIDTH_M) / 2),
    "left_six_yard_bottom": (GOAL_BOX_LENGTH_M, (PITCH_WIDTH_M + GOAL_BOX_WIDTH_M) / 2),
    "right_six_yard_top": (PITCH_LENGTH_M - GOAL_BOX_LENGTH_M, (PITCH_WIDTH_M - GOAL_BOX_WIDTH_M) / 2),
    "right_six_yard_bottom": (PITCH_LENGTH_M - GOAL_BOX_LENGTH_M, (PITCH_WIDTH_M + GOAL_BOX_WIDTH_M) / 2),
    # Goal posts. Click the point where each post meets the grass (ground
    # level) - the crossbar/post top is *not* on the pitch plane and will
    # bias the homography if used as a correspondence.
    "left_goal_post_top": (0.0, (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2),
    "left_goal_post_bottom": (0.0, (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2),
    "right_goal_post_top": (PITCH_LENGTH_M, (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2),
    "right_goal_post_bottom": (PITCH_LENGTH_M, (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2),
    # Centre-circle tangent points on the halfway line and long axis.
    "centre_circle_top": (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2 - CENTRE_CIRCLE_RADIUS_M),
    "centre_circle_bottom": (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2 + CENTRE_CIRCLE_RADIUS_M),
    "centre_circle_left": (PITCH_LENGTH_M / 2 - CENTRE_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2),
    "centre_circle_right": (PITCH_LENGTH_M / 2 + CENTRE_CIRCLE_RADIUS_M, PITCH_WIDTH_M / 2),
}

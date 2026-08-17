from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .calibration import PitchCalibrator
from .pitch_keypoints import PitchKeypointDetector
from .team import (
    DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES,
    TeamClassifier,
    resolve_goalkeeper_team_ids,
    resolve_track_team_ids,
)
from .tracking import PlayerTracker


@dataclass
class TrackingPipeline:
    """Runs detection+tracking over a video and projects every tracked
    object's pitch-plane anchor point onto metric pitch coordinates via a
    fitted `PitchCalibrator`.

    If a fitted `team_classifier` is supplied, detections whose class name is
    in `team_eligible_class_names` (default: "person" for the generic COCO
    model, "player" for the specialized football-player-detection.pt model)
    get a `team_id`, resolved to one stable label per track (see
    `pitchvision.team.resolve_track_team_ids`). If the tracker's detections
    include a "goalkeeper" class (the specialized model), goalkeeper tracks
    are then assigned a `team_id` by proximity to each team's centroid
    instead of colour (`pitchvision.team.resolve_goalkeeper_team_ids`) - a
    no-op otherwise. A "referee" class, if present, is never assigned a
    `team_id` and is left out of team-level analysis entirely.

    `calibrator` is the homography used from the start of the clip, and the
    one kept for the rest of it if `keypoint_detector` is left unset. Pass a
    `keypoint_detector` to instead re-run automatic pitch-keypoint detection
    every `recalibration_interval` frames and refit the homography from
    scratch each time - broadcast footage often pans/zooms within a clip
    (build-up-phase clips especially), so a single homography fit once at
    the start silently drifts as the frame moves away from the one it was
    calibrated on. Each row records which frame's homography it was
    projected with (`calibration_frame`), so the reliability of the pitch
    coordinates over time can be checked afterwards. Whenever a
    recalibration attempt can't find enough confident keypoints (occlusion,
    a mid-pan blur), the previous homography is kept rather than the run
    failing.

    The output DataFrame is the shared input the three phase-of-play
    analyses (goal-scoring opportunity, build-up, set pieces) build on.
    """

    tracker: PlayerTracker
    calibrator: PitchCalibrator
    team_classifier: Optional[TeamClassifier] = None
    team_eligible_class_names: Iterable[str] = DEFAULT_TEAM_ELIGIBLE_CLASS_NAMES
    keypoint_detector: Optional[PitchKeypointDetector] = None
    recalibration_interval: int = 30

    def run(self, video_path: str, max_frames: Optional[int] = None) -> pd.DataFrame:
        eligible_class_names = set(self.team_eligible_class_names)
        calibrator = self.calibrator
        calibration_frame = 0
        rows = []
        for frame_idx, (frame, objects) in enumerate(self.tracker.track_video(video_path)):
            if max_frames is not None and frame_idx >= max_frames:
                break
            if self.keypoint_detector is not None and frame_idx % self.recalibration_interval == 0:
                try:
                    calibrator = self.keypoint_detector.calibrate(frame)
                    calibration_frame = frame_idx
                except RuntimeError:
                    pass  # too few confident keypoints this frame - keep the last good homography
            for obj in objects:
                pitch_x, pitch_y = calibrator.pixel_to_pitch(obj.foot_point)[0]
                team_id = None
                if self.team_classifier is not None and obj.class_name in eligible_class_names:
                    team_id = self.team_classifier.predict(frame, obj.xyxy)
                x1, y1, x2, y2 = obj.xyxy
                rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": obj.track_id,
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "confidence": obj.confidence,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "pixel_x": obj.foot_point[0],
                        "pixel_y": obj.foot_point[1],
                        "pitch_x": pitch_x,
                        "pitch_y": pitch_y,
                        "team_id": team_id,
                        "calibration_frame": calibration_frame,
                    }
                )
        df = pd.DataFrame(rows)
        if self.team_classifier is not None and not df.empty:
            df = resolve_track_team_ids(df, class_names=eligible_class_names)
            df = resolve_goalkeeper_team_ids(df)
        return df

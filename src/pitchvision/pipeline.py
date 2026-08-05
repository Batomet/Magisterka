from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .calibration import PitchCalibrator
from .team import TeamClassifier, resolve_track_team_ids
from .tracking import PlayerTracker


@dataclass
class TrackingPipeline:
    """Runs detection+tracking over a video and projects every tracked
    object's pitch-plane anchor point onto metric pitch coordinates via a
    fitted `PitchCalibrator`. If a fitted `team_classifier` is supplied,
    each person detection also gets a `team_id`, resolved to one stable
    label per track (see `pitchvision.team.resolve_track_team_ids`).

    The output DataFrame is the shared input the three phase-of-play
    analyses (goal-scoring opportunity, build-up, set pieces) build on.
    """

    tracker: PlayerTracker
    calibrator: PitchCalibrator
    team_classifier: Optional[TeamClassifier] = None

    def run(self, video_path: str, max_frames: Optional[int] = None) -> pd.DataFrame:
        rows = []
        for frame_idx, (frame, objects) in enumerate(self.tracker.track_video(video_path)):
            if max_frames is not None and frame_idx >= max_frames:
                break
            for obj in objects:
                pitch_x, pitch_y = self.calibrator.pixel_to_pitch(obj.foot_point)[0]
                team_id = None
                if self.team_classifier is not None and obj.class_name == "person":
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
                    }
                )
        df = pd.DataFrame(rows)
        if self.team_classifier is not None and not df.empty:
            df = resolve_track_team_ids(df)
        return df

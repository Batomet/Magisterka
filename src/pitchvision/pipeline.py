from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .calibration import PitchCalibrator
from .tracking import PlayerTracker


@dataclass
class TrackingPipeline:
    """Runs detection+tracking over a video and projects every tracked
    object's pitch-plane anchor point onto metric pitch coordinates via a
    fitted `PitchCalibrator`.

    The output DataFrame is the shared input the three phase-of-play
    analyses (goal-scoring opportunity, build-up, set pieces) build on.
    """

    tracker: PlayerTracker
    calibrator: PitchCalibrator

    def run(self, video_path: str, max_frames: Optional[int] = None) -> pd.DataFrame:
        rows = []
        for frame_idx, objects in enumerate(self.tracker.track_video(video_path)):
            if max_frames is not None and frame_idx >= max_frames:
                break
            for obj in objects:
                pitch_x, pitch_y = self.calibrator.pixel_to_pitch(obj.foot_point)[0]
                rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": obj.track_id,
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "confidence": obj.confidence,
                        "pixel_x": obj.foot_point[0],
                        "pixel_y": obj.foot_point[1],
                        "pitch_x": pitch_x,
                        "pitch_y": pitch_y,
                    }
                )
        return pd.DataFrame(rows)

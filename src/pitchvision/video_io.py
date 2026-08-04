import glob
import os
from typing import Iterator, List

import cv2
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def mount_drive() -> None:
    """Mount Google Drive in a Colab runtime. No-op outside Colab."""
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.mount("/content/drive")


def list_videos(folder: str) -> List[str]:
    """Returns all video files directly inside `folder`, sorted by name."""
    paths = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


class VideoFrames:
    """Iterates over a video file's frames as BGR numpy arrays."""

    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[np.ndarray]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def read_frame(self, index: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise IndexError(f"Frame {index} out of range for {self.path}")
        return frame

    def release(self) -> None:
        self._cap.release()

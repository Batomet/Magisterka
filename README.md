# Magisterka

**Using Projective Transformation for the Spatial Analysis of Team Behaviors in Football: A Study of Selected Phases of Play**

Automation of spatial data acquisition from football match video for tactical
analysis. Players and the ball are detected with YOLO and tracked with
ByteTrack; an automatic pitch-plane calibration module (homographic
transformation) maps pixel coordinates from the broadcast/perspective view to
metric 2D pitch coordinates (top-down view). That common pipeline feeds a
quantitative analysis of team spatial structure across three phases of play:
goal-scoring opportunities (defensive compactness, Voronoi space control),
build-up in midfield (Effective Playing Space via Convex Hull, formation
centroids), and set pieces / breaks in play (positional-structure
repeatability, static-to-dynamic transition timing).

## Repository layout

```
src/pitchvision/       Core Python package
  config.py             Pitch dimensions, COCO class ids, Drive folder config
  video_io.py            Drive mounting, video listing, frame iteration
  detection.py           YOLOv8 wrapper (player + ball detection)
  tracking.py             ByteTrack-based multi-object tracking
  calibration.py          Homography pitch calibration (pixel <-> metric pitch coords)
  pitch.py                 Top-down pitch drawing (matplotlib)
  pipeline.py              Ties detection+tracking+calibration into a track DataFrame
notebooks/
  00_pipeline_demo.ipynb   Colab notebook: run the core pipeline end-to-end on a sample clip
```

## Status

Implemented: detection, tracking, and pitch calibration core (this is the
shared foundation all three phase analyses build on).

Not yet implemented: team classification, and the three phase-specific
analyses themselves (goal-scoring opportunity, build-up, set pieces) —
see the "Next steps" cell at the end of `00_pipeline_demo.ipynb`.

## Usage (Google Colab)

1. Footage lives in three Google Drive folders: `BuildingAction`, `Goals`,
   `SetPieces`.
2. Open `notebooks/00_pipeline_demo.ipynb` in Colab (or run it via the
   `File > Open notebook > GitHub` dialog pointed at this repo/branch).
3. Run the cells top to bottom: they clone this repo, install dependencies,
   mount Drive, sanity-check detection, calibrate the pitch homography from a
   handful of manually identified landmarks, run the full tracking pipeline,
   and save the resulting per-frame pitch-coordinate tracks as a CSV back to
   Drive.

## Local development

```bash
pip install -e .
```

Ball detection uses the COCO `sports ball` class from a generic pretrained
YOLOv8 checkpoint; small, fast-moving broadcast footballs are hard for a
generic model to pick up reliably; swap in a football-specific fine-tuned
checkpoint (`PlayerBallDetector(weights=...)` / `PlayerTracker(weights=...)`)
once one is trained.

Pitch calibration is point-correspondence based: you supply >=4 pixel<->pitch
landmark matches per clip (see `pitchvision.calibration.PITCH_LANDMARKS_M`)
and `PitchCalibrator.from_point_pairs` fits the homography via
`cv2.findHomography`. A fully automatic keypoint-detection-based calibrator
can later be substituted as a drop-in, since it only needs to produce the
same `(pixel_points, pitch_points)` arrays.

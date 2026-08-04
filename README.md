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
  pitch_keypoints.py       Automatic calibration via a pretrained pitch-keypoint model
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
   mount Drive, sanity-check detection, calibrate the pitch homography
   automatically from detected pitch keypoints (manual point-picking is
   available as a fallback), run the full tracking pipeline, and save the
   resulting per-frame pitch-coordinate tracks as a CSV back to Drive.

## Local development

```bash
pip install -e .
```

Ball detection uses the COCO `sports ball` class from a generic pretrained
YOLOv8 checkpoint; small, fast-moving broadcast footballs are hard for a
generic model to pick up reliably; swap in a football-specific fine-tuned
checkpoint (`PlayerBallDetector(weights=...)` / `PlayerTracker(weights=...)`)
once one is trained.

Pitch calibration is homography-based (`PitchCalibrator`, `cv2.findHomography`
under the hood), fit from >=4 pixel<->pitch point correspondences. Two ways
to get those correspondences:

- **Automatic (`pitch_keypoints.py`)**: `PitchKeypointDetector` runs a
  pretrained YOLOv8-pose model (`football-pitch-detection.pt`, from
  [roboflow/sports](https://github.com/roboflow/sports), downloaded once via
  `download_pitch_keypoint_weights` — public Google Drive link, no account
  needed) that localises the 32 standard pitch keypoints directly in a frame.
  The confidently-detected ones are matched against `pitch_keypoint_template()`
  and fed straight into `PitchCalibrator.from_point_pairs`. Note: that
  upstream project's own template uses a non-standard pitch scale (e.g. a
  20.15 m penalty-box depth, when the Laws of the Game fix that at 16.5 m
  regardless of pitch size) — `pitch_keypoint_template()` reuses only its
  keypoint *ordering*, recomputing every coordinate from the real,
  Laws-of-the-Game-accurate constants in `config.py`.
- **Manual fallback (`calibration.PITCH_LANDMARKS_M`)**: supply pixel<->pitch
  matches by hand for clips where automatic detection can't find enough
  confident keypoints (heavy occlusion, an unusual crop).

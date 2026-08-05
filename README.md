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
  config.py             Pitch dimensions, detector class ids, Drive folder config
  video_io.py            Drive mounting, video listing, frame iteration
  detection.py           YOLOv8 wrapper (player + ball detection); specialized 4-class weights download
  _weights.py             Shared public-Google-Drive checkpoint download helper
  tracking.py             ByteTrack-based multi-object tracking
  calibration.py          Homography pitch calibration (pixel <-> metric pitch coords)
  pitch_keypoints.py       Automatic calibration via a pretrained pitch-keypoint model
  pitch.py                 Top-down pitch drawing (matplotlib)
  team.py                  Jersey-colour team classification (KMeans)
  pipeline.py              Ties detection+tracking+calibration+team classification into a track DataFrame
notebooks/
  00_pipeline_demo.ipynb   Colab notebook: run the core pipeline end-to-end on a sample clip
```

## Status

Implemented: detection, tracking, pitch calibration, and team classification
— the shared foundation all three phase analyses build on.

Not yet implemented: the three phase-specific analyses themselves
(goal-scoring opportunity, build-up, set pieces) — see the "Next steps" cell
at the end of `00_pipeline_demo.ipynb`.

## Usage (Google Colab)

1. Footage lives in three Google Drive folders: `BuildingAction`, `Goals`,
   `SetPieces`.
2. Open `notebooks/00_pipeline_demo.ipynb` in Colab (or run it via the
   `File > Open notebook > GitHub` dialog pointed at this repo/branch).
3. Run the cells top to bottom: they clone this repo, install dependencies,
   mount Drive, sanity-check detection, calibrate the pitch homography
   automatically from detected pitch keypoints (manual point-picking is
   available as a fallback), cluster players into two teams by jersey
   colour, run the full tracking pipeline with team labels attached, and
   save the resulting per-frame pitch-coordinate tracks as a CSV back to
   Drive.

## Local development

```bash
pip install -e .
```

`PlayerBallDetector`/`PlayerTracker` default to a generic COCO-pretrained
YOLOv8 checkpoint (`yolov8n.pt`; class 0 = person, class 32 = sports ball) -
simple, no extra download, but it can't tell players/goalkeepers/referees
apart and is unreliable on the ball. For better detection and team
classification, point `weights` at the specialized
`football-player-detection.pt` checkpoint instead (also from
[roboflow/sports](https://github.com/roboflow/sports); a 4-class model: ball,
goalkeeper, player, referee), downloaded once via
`download_player_detection_weights` — same public-Google-Drive, no-account
pattern as the pitch-keypoint weights — and pass
`classes=SPORTS_DETECTION_CLASSES` (its class ids are unrelated to the COCO
ones).

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

Team classification (`team.py`) is a classical colour-clustering approach:
`collect_jersey_colors` samples player crops across a clip and reduces each
to a robust (hue, saturation) signature (torso region only, pitch-grass
pixels masked out, brightness/value ignored since it swings with shadows);
`TeamClassifier` fits a 2-cluster KMeans over those signatures, and
`TrackingPipeline` (when given a fitted classifier) predicts a `team_id` for
detections whose class is in `team_eligible_class_names`, then collapses
each track's noisy per-frame predictions to one stable majority-vote label
via `resolve_track_team_ids`. It's lightweight — no extra model beyond the
detector checkpoint, unlike embedding-based classifiers used elsewhere in
the sports-analytics community — but cluster ids (`0`/`1`) still aren't
team-identity aware; check `TeamClassifier.cluster_swatches` to see which is
which.

Referees and goalkeepers need the specialized player detector above to
resolve properly, since jersey-colour clustering alone can't separate them
from outfield players:
- With the generic COCO model (everything is class `person`), a referee's or
  goalkeeper's kit colour just gets assigned to whichever of the 2 clusters
  is nearest — wrong more often than not, since both conventionally wear a
  third, distinct colour. Workarounds: set `n_clusters=3` and treat the
  extra cluster as "other", or manually reassign known referee/goalkeeper
  track_ids after classification.
- With the specialized model (classes `player`/`goalkeeper`/`referee`/`ball`),
  fit `TeamClassifier` only on `class_name == "player"` samples (the default
  `collect_jersey_colors` behaviour) so referees never enter the colour fit
  at all and keep `team_id = None` throughout. Goalkeepers are then resolved
  by `resolve_goalkeeper_team_ids` — nearest team centroid by pitch position,
  not colour — since `TrackingPipeline.run` calls it automatically after
  `resolve_track_team_ids` whenever a `goalkeeper` class is present (a no-op
  otherwise).

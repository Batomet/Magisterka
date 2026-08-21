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
  compactness.py           Inter-player-distance/centroid compactness metrics (generic, reused across phases)
  voronoi.py                Voronoi-tessellation space control (generic, reused across phases)
  convex_hull.py             Effective Playing Space via Convex Hull (generic, reused across phases)
  set_pieces.py               Restart detection, transition timing, cross-clip formation repeatability
  pipeline.py              Ties detection+tracking+calibration+team classification into a track DataFrame
  evaluation.py            Validation metrics: detection precision/recall/F1, calibration holdout error, clustering accuracy
notebooks/
  00_pipeline_demo.ipynb                 Colab notebook: run the core pipeline end-to-end on a sample clip
  01_goal_scoring_opportunity.ipynb       Defensive compactness + Voronoi space control, batch-processed over every Goals clip
  02_build_up_phase.ipynb                 Effective Playing Space + formation stretching + Voronoi space control, batch-processed over every BuildingAction clip
  03_set_pieces.ipynb                     Transition timing + formation repeatability across multiple SetPieces clips
  04_validation.ipynb                     Detection/calibration/clustering error checks, batch-processed over every clip in all three folders (automatic by default, optional hand-labeled sections for precise numbers)
```

## Status

All three phase-specific analyses from the abstract are implemented, on top
of the shared foundation (detection, tracking, pitch calibration, team
classification):

- Goal-scoring opportunity: defensive compactness + Voronoi space control
  (`01_goal_scoring_opportunity.ipynb`).
- Build-up: Effective Playing Space via Convex Hull + formation-centroid
  stretching + Voronoi space control (`02_build_up_phase.ipynb`).
- Set pieces / breaks in play: static-to-dynamic transition timing +
  cross-clip positional-structure repeatability (`03_set_pieces.ipynb`).

## Usage (Google Colab)

1. Footage lives in three Google Drive folders: `BuildingAction`, `Goals`,
   `SetPieces`.
2. Start with `notebooks/00_pipeline_demo.ipynb` in Colab (or run it via the
   `File > Open notebook > GitHub` dialog pointed at this repo/branch) to
   sanity-check the pipeline on a clip: detection, automatic pitch
   calibration (manual point-picking as a fallback), team classification,
   and the full tracking pipeline, saved as a CSV back to Drive.
3. Then open whichever phase-specific notebook(s) you need:
   `notebooks/01_goal_scoring_opportunity.ipynb` (defensive compactness +
   Voronoi space control) and `notebooks/02_build_up_phase.ipynb` (Effective
   Playing Space + formation stretching + Voronoi space control) each
   batch-process **every** clip in `Goals`/`BuildingAction` respectively -
   one pass over the whole folder,
   with a per-clip failure (bad calibration, too few jersey samples) skipped
   rather than stopping the run, and results saved both per-clip and as one
   combined `*_all_clips_summary.csv`. A separate "inspect one clip closely"
   section at the end of each notebook re-uses the batch results (no
   re-running the pipeline) for a close look - including, in
   `01_goal_scoring_opportunity.ipynb`, hand-identifying which team was
   defending, since that can't be automated. `notebooks/03_set_pieces.ipynb`
   instead needs *several* clips of the *same* restart type, hand-picked by
   watching them first (`instance_videos`) - repeatability across genuinely
   different restart types wouldn't be a meaningful comparison, so this one
   is deliberately not "every clip in the folder". Each notebook re-runs the
   shared setup condensed into one section before its own analysis.

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
ones). Also pass `imgsz=1280`: Ultralytics' own default (640) downscales a
broadcast frame significantly before inference, which can drop small,
clustered, or partially-occluded players below what the model can reliably
pick up (e.g. a crowd of players in the box) - the checkpoint's own
reference implementation (roboflow/sports) runs it at 1280, not 640.

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

By default `TrackingPipeline` reuses one homography (fit once, before the
run starts) for the whole clip, which is only correct if the camera doesn't
pan/zoom/cut. Broadcast footage often does move within a clip - pass a
`keypoint_detector` (a fitted `PitchKeypointDetector`) to `TrackingPipeline`
instead, and it re-runs automatic keypoint detection and refits the
homography every `recalibration_interval` frames (default 30) rather than
just once. If a recalibration attempt can't find enough confident keypoints
(occlusion, a mid-pan blur), the previous homography is kept for that frame
rather than the run failing. Each output row records `calibration_frame` -
the frame index of the homography actually used to project it - so
recalibration health can be checked afterwards (e.g.
`tracks_df["calibration_frame"].value_counts()`; long runs stuck on one
value mean recalibration kept failing and the pipeline fell back
throughout).

Team classification (`team.py`) is a classical colour-clustering approach:
`collect_jersey_colors` samples player crops across a clip and reduces each
to a 3-feature signature via `extract_jersey_color` - median hue, median
saturation (over a *tight, central* torso crop - see `_torso_crop`: a
vertical band skipping the head/neck and everything below the chest, trimmed
on both horizontal edges - with any remaining pitch-grass-coloured pixels
masked out as a secondary safety net), and the *standard deviation* of
value/brightness. The crop is deliberately tight rather than relying mainly
on the grass-colour mask: detector boxes aren't pixel-tight, and on some
pitches/lighting the grass renders yellowish-green - close enough in hue to
a yellow kit that no hue-range filter can reliably tell them apart, so
physically cropping away the background matters more than filtering it out
after the fact. `top`/`bottom`/`horizontal_margin` are exposed as keyword
arguments through `extract_jersey_color`, `collect_jersey_samples`, and
`collect_jersey_colors` (as `**crop_kwargs`) for tuning per clip - use
`plot_jersey_color_samples` (below) to see whether the defaults are still
picking up background for a given clip's box tightness/camera distance.
Median hue/saturation alone also breaks down for a monochrome or striped kit
(e.g. black/white): near-black and near-white pixels both have near-zero
saturation and essentially undefined hue, so that team's colour signal is
weak - the value-std feature captures "how patterned/high-contrast is this
kit" instead, which a solid-coloured kit doesn't have. `TeamClassifier`
standardises all three features to zero mean/unit variance
(`sklearn.preprocessing.StandardScaler`) before fitting a 2-cluster KMeans,
so no single feature's numeric scale dominates the distance metric, and
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

  `resolve_track_team_ids` only ever stamps its majority-vote `team_id` onto
  a row whose OWN `class_name` that frame is eligible (e.g. `"player"`) - a
  track's majority is computed from its eligible frames, but never applied
  to that same track's frames the detector itself called something else
  that frame. This matters because the detector's class prediction can
  flicker frame-to-frame for the same tracked object: without this
  restriction, a referee briefly (mis)classified `"player"` in even one
  frame would leak a real `team_id` onto every frame of that referee's
  track, including the ones correctly labeled `"referee"`. It's still not
  perfect - the one flickered frame itself still gets a forced nearest-cluster
  `team_id`, since nothing here can tell that specific frame's class
  prediction was wrong - just the ones that don't flicker.

**Audit a fit before trusting it.** KMeans with `n_clusters=2` doesn't
guarantee the split lands on team identity - it splits along whatever axis
has the most variance in the sampled colours, which can just as easily be
lighting, motion blur, or grass bleeding into a loose crop, especially if
intra-team variance (from crop quality) is comparable to or larger than the
real inter-team colour difference. Cluster sizes and swatch colours alone
can't tell trustworthy apart from spurious: `collect_jersey_samples` (like
`collect_jersey_colors`, but also keeps each sample's actual crop) plus
`plot_jersey_color_samples` (a (hue, saturation) scatter coloured by cluster
assignment, alongside a grid of the actual crops used, bordered by their
assigned cluster) makes the difference visible - two separated blobs of
roughly similar size versus one tight blob plus a handful of scattered
outliers.

Three generic spatial-analysis primitives, deliberately not tied to one
phase - `compactness.py` and `voronoi.py` are shared by the goal-scoring
opportunity analysis, `compactness.py` and `convex_hull.py` by build-up:

- **`compactness.py`**: `compute_frame_compactness` takes one team's (x, y)
  positions in a single frame and returns mean pairwise inter-player
  distance, "stretch index" (mean distance from the team centroid), and the
  formation's length/width extent. `compute_team_compactness` applies that
  across every frame of a tracked clip for one `team_id`, returning a
  per-frame time series DataFrame - goalkeepers excluded by default, since
  they'd distort an outfield defensive-line metric. `compute_centroid_separation`
  complements that with the distance *between* two teams' centroids per
  frame (length-axis/width-axis/overall) - a team's own stretch describes
  its internal shape, this describes how far it's been pulled from the
  opposition.
- **`voronoi.py`**: `pitch_voronoi_cells` tessellates the pitch by a set of
  player positions (`scipy.spatial.Voronoi`, with dummy points added far
  outside the pitch so every region comes out bounded, then each cell is
  clipped to the actual pitch rectangle via `shapely`) - the standard
  simplified "space control" model in tactical analysis: the pitch area
  closer to a player than to anyone else is credited to them. It ignores
  player speed/orientation/reaction time, unlike more advanced pitch-control
  models, but is a well-established first-order approximation.
  `compute_space_control` applies this across every frame of a tracked clip,
  returning each team's total controlled area (m²) per frame; `plot_voronoi`
  draws the cells onto a `draw_pitch()` axes, coloured by team;
  `save_voronoi_frames` batch-renders one PNG per frame to a folder.
- **`convex_hull.py`**: `convex_hull_polygon` (via `shapely.geometry.MultiPoint`)
  is the "Effective Playing Space" primitive - the smallest polygon
  containing a set of positions. `compute_team_convex_hull` gives one
  team's hull area/perimeter per frame (goalkeepers excluded by default,
  same reasoning as `compactness.py`); `compute_combined_convex_hull` does
  the same over *both* teams' players together, the classical Frencken et
  al. (2011) "Effective Playing Space" definition - it can shrink even
  while one team's own hull grows, if both teams move into overlapping
  space rather than spreading apart, so the per-team and combined figures
  answer different questions. `plot_convex_hull` draws a hull outline onto
  a `draw_pitch()` axes; `save_convex_hull_frames` batch-renders one PNG per
  frame to a folder, mirroring `voronoi.save_voronoi_frames`.

All of these modules' DataFrame-level functions work on *any* DataFrame
shaped like `TrackingPipeline.run()`'s output (`frame`, `class_name`,
`team_id`, `pitch_x`, `pitch_y` columns at minimum) - a CSV you build by
hand, or reload from a previous run via `pd.read_csv(...)`, works
identically to a live pipeline result, no detection/tracking/calibration
required to regenerate diagrams or recompute metrics from data you already
have.

**`set_pieces.py`** covers the third phase - unlike the modules above, its
two analyses are genuinely specific to breaks in play rather than reusable
elsewhere:

- **Static-to-dynamic transition timing**: `compute_track_speeds` adds a
  `speed_mps` column to a tracks DataFrame - each track's frame-to-frame
  displacement divided by actual elapsed time, computed from *smoothed*
  positions (a short centred rolling mean) purely for the speed calculation,
  since a few centimetres of realistic tracking jitter on a stationary
  player can otherwise look like several m/s of "speed" once pushed through
  the homography (verified numerically before relying on it: unsmoothed
  jitter alone produced spikes over 4 m/s for a genuinely motionless
  player, comfortably above the default "moving" threshold - smoothing
  brought that safely under it). `detect_restart_frame` finds the restart
  of play from the ball's movement onset (sustained speed above a
  threshold, not a single noisy detection - needs the specialized
  detector's `"ball"` class); `detect_team_dynamic_frame` finds when a
  sustained fraction of a team's players start moving; `compute_transition_time`
  combines both into a restart→dynamic time delta, returning `None`
  (rather than a misleading zero) if the ball was never confidently
  tracked in a clip. `plot_team_speed_timeline` visualises a team's mean
  speed over the clip with both detected frames marked, to sanity-check the
  automatic detection against what's actually in the video.
- **Repeatability of positional structures**: a player's `track_id` doesn't
  carry over between separate set-piece clips (no persistent
  re-identification), so comparing formations across instances can't rely
  on matching specific players by identity. `extract_static_formation`
  takes a team's average player positions over a short window before a
  clip's restart frame; `match_formations` finds the *optimal* one-to-one
  pairing between two such position sets (the Hungarian algorithm,
  `scipy.optimize.linear_sum_assignment`, minimising total assignment
  distance) and reports how far apart that best-case pairing still is;
  `compute_formation_repeatability` averages that pairwise distance across
  every pair of instances - lower means a more consistent structure from
  one restart to the next. Since `TeamClassifier` is fit independently per
  clip (cluster id `0`/`1` isn't guaranteed to mean the same real team
  across clips), `pick_team_by_swatch_color` resolves "which cluster id is
  the team I'm tracking" in each clip by nearest colour match to a
  reference RGB, rather than assuming id `0` always means the same team.
  `plot_formations` overlays multiple instances' formations on one
  `draw_pitch()` axes for a visual check alongside the numeric score.

All of the above was verified against known/synthetic inputs before being
wired into a notebook - exact run-length detection on hand-built boolean
sequences, exact restart/transition frames on synthetic stationary-then-moving
tracks, zero assignment distance for identical point sets under reordering,
lower repeatability distance for near-identical formations than for
substantially different ones, and the jitter/smoothing effect described
above - since this module's logic (run-length detection over noisy speed
signals, optimal assignment) is considerably less obvious to get right by
inspection than the more direct geometric computations in the other
modules.

**`evaluation.py`** quantifies error in the three places it actually enters the
pipeline, for the thesis's validation/methodology section - none of them has a
pre-existing ground-truth split for this project's own broadcast footage.
`notebooks/04_validation.ipynb` runs top to bottom with no manual input by
default, same as `00`-`03`, over **every clip in `Goals`/`BuildingAction`/`SetPieces`**
- results are saved to one accumulating `validation_all_clips.csv` on Drive:
re-running the notebook (e.g. after adding new clips) merges in rather than
overwriting, with a re-validated clip's row replaced rather than duplicated
(matched on clip name + source folder). Calibration's holdout error is
computed from
`PitchKeypointDetector`'s own automatically-detected keypoints (their true
pitch position is a known Laws-of-the-Game constant, not something a human
needs to label, so no manual point-picking is needed at all); detection and
clustering fall back to automatic, ground-truth-free proxy checks (per-class
detection count/confidence stability across the clip; a silhouette score for
how well-separated the two colour clusters are) since there's no way to
compute real precision/recall or clustering accuracy without a human saying
what's actually true. Optional `RUN_MANUAL_.../USE_MANUAL_...` flags (off by
default, same pattern as notebook `00`'s `USE_MANUAL_FALLBACK`) turn on real
hand-labeled precision/recall and clustering-accuracy numbers when a more
precise figure is worth the few minutes of hand-labeling - hover over a
`plotly` frame display to read pixel coordinates, then fill them into a dict:

- **Calibration (homography)** - automatic by default: `PitchCalibrator.reprojection_error`
  alone measures error on the *same* points a homography was fit from, which
  always looks good and says nothing about accuracy elsewhere on the pitch.
  `compute_calibration_holdout_error` instead does repeated random
  subsampling (Monte Carlo) validation: given more point correspondences than
  the minimum 4 a fit needs - by default `PitchKeypointDetector`'s own
  confidently-detected automatic keypoints, or hand-picked landmarks via the
  optional manual fallback - it repeatedly fits on a random subset and
  measures reprojection error, in metres, on the rest, pooling every
  held-out error across many random splits into one mean/std/max.
- **Detection (YOLO)** - optional manual section for a real number:
  `predicted_boxes_dataframe`/`ground_truth_boxes_dataframe` build matching
  DataFrames from the detector's own output on a handful of chosen frames and
  from hand-labeled boxes for those same frames; `compute_detection_metrics`
  matches them by IoU per (frame, class) - greedy highest-IoU-first,
  class-scoped so a correctly-placed but mislabeled box counts as both a
  false positive for its predicted class and a false negative for its true
  one - and returns precision/recall/F1/mean-IoU per class plus a
  micro-averaged "overall" row.
- **Team clustering (K-means)** - optional manual section for a real number:
  `compute_clustering_accuracy` compares hand-labeled true team per track
  against the resolved cluster `team_id`, under the *optimal*
  cluster-id-to-true-label matching (the Hungarian algorithm on the
  confusion matrix, `scipy.optimize.linear_sum_assignment` - the same tool
  `set_pieces.match_formations` uses for a different optimal
  pairing problem) - required because a cluster id (`0`/`1`) carries no
  identity of its own.

All three metric functions were verified against synthetic inputs (exact IoU
on known box overlaps, near-zero holdout error on an exactly-homography-able
synthetic point set that grows with injected pixel noise, and exact accuracy
recovery for a clustering that's a known permutation of the true labels)
before being wired into the notebook.

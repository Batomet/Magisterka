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
  pipeline.py              Ties detection+tracking+calibration+team classification into a track DataFrame
notebooks/
  00_pipeline_demo.ipynb                 Colab notebook: run the core pipeline end-to-end on a sample clip
  01_goal_scoring_opportunity.ipynb       Defensive compactness + Voronoi space control on a Goals clip
  02_build_up_phase.ipynb                 Effective Playing Space + formation stretching on a BuildingAction clip
```

## Status

Implemented: detection, tracking, pitch calibration, team classification —
the shared foundation all three phase analyses build on — plus two of the
three phase-specific analyses: goal-scoring opportunity (defensive
compactness + Voronoi space control, `01_goal_scoring_opportunity.ipynb`)
and build-up (Effective Playing Space via Convex Hull + formation-centroid
stretching, `02_build_up_phase.ipynb`).

Not yet implemented: set pieces / breaks in play (positional-structure
repeatability, static-to-dynamic transition timing) — see the "Next steps"
cell at the end of `01_goal_scoring_opportunity.ipynb`.

## Usage (Google Colab)

1. Footage lives in three Google Drive folders: `BuildingAction`, `Goals`,
   `SetPieces`.
2. Start with `notebooks/00_pipeline_demo.ipynb` in Colab (or run it via the
   `File > Open notebook > GitHub` dialog pointed at this repo/branch) to
   sanity-check the pipeline on a clip: detection, automatic pitch
   calibration (manual point-picking as a fallback), team classification,
   and the full tracking pipeline, saved as a CSV back to Drive.
3. Then open `notebooks/01_goal_scoring_opportunity.ipynb` (defensive
   compactness + Voronoi space control on a `Goals` clip) and/or
   `notebooks/02_build_up_phase.ipynb` (Effective Playing Space + formation
   stretching on a `BuildingAction` clip) - each re-runs the same setup
   condensed into one section, applies its own analysis, and saves the
   results as CSVs.

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

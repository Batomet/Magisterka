from dataclasses import dataclass

# Standard FIFA-regulation pitch dimensions in metres (length x width).
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Laws-of-the-Game pitch markings, in metres. Unlike overall pitch length/width,
# these are fixed regardless of the specific pitch's size.
PENALTY_BOX_LENGTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
GOAL_BOX_LENGTH_M = 5.5
GOAL_BOX_WIDTH_M = 18.32
CENTRE_CIRCLE_RADIUS_M = 9.15
PENALTY_SPOT_DISTANCE_M = 11.0
GOAL_WIDTH_M = 7.32

# COCO class ids used for detection: 0 = person, 32 = sports ball.
COCO_PERSON_CLASS_ID = 0
COCO_BALL_CLASS_ID = 32
DETECTION_CLASSES = (COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID)

# Class ids for the specialized `football-player-detection.pt` checkpoint
# (see pitchvision.detection.download_player_detection_weights) - a 4-class
# model trained specifically to tell players, goalkeepers, and referees
# apart, unlike the generic COCO `person` class above. These ids are unrelated
# to the COCO ones and only apply when a detector/tracker is pointed at that
# checkpoint's weights.
SPORTS_BALL_CLASS_ID = 0
SPORTS_GOALKEEPER_CLASS_ID = 1
SPORTS_PLAYER_CLASS_ID = 2
SPORTS_REFEREE_CLASS_ID = 3
SPORTS_DETECTION_CLASSES = (
    SPORTS_BALL_CLASS_ID,
    SPORTS_GOALKEEPER_CLASS_ID,
    SPORTS_PLAYER_CLASS_ID,
    SPORTS_REFEREE_CLASS_ID,
)


@dataclass
class DriveConfig:
    """Paths to the three Google Drive footage folders, relative to a root."""

    root: str = "/content/drive/MyDrive"
    building_action_dir: str = "BuildingAction"
    goals_dir: str = "Goals"
    set_pieces_dir: str = "SetPieces"

    @property
    def building_action_path(self) -> str:
        return f"{self.root}/{self.building_action_dir}"

    @property
    def goals_path(self) -> str:
        return f"{self.root}/{self.goals_dir}"

    @property
    def set_pieces_path(self) -> str:
        return f"{self.root}/{self.set_pieces_dir}"

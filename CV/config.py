"""
Configuration file for Detection and Tracking
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "test_video/IMG_1251.MOV"
OUTPUT_PATH = PROJECT_ROOT / "output"

# Model settings
MODEL_CONFIDENCE = 0.25
MODEL_IOU_THRESHOLD = 0.45

# Tracking settings
TRACK_ACTIVATION_THRESHOLD = 0.25
TRACK_LOST_DELAY = 30
TRACK_FRAME_RATE = 30

# No counting line settings needed

# Hygiene product detection settings
DETECT_HANDS = True      # Whether to detect hands
DETECT_PRODUCTS = True   # Whether to detect hygiene products
MIN_DETECTION_SIZE = 50  # Minimum size for detection (pixels)

# Video processing settings
FRAME_SKIP = 1  # Process every Nth frame
OUTPUT_FPS = 30

# Detection classes (hygiene products)
HYGIENE_CLASSES = [
    'background', 'hand', 'packet', 'tampon', 'pad'
]

# Visualization settings
BOX_THICKNESS = 2
TEXT_THICKNESS = 2
TEXT_SCALE = 1.0
TRACK_LENGTH = 30  # Number of frames to show track history

# Trajectory analysis settings for grabbed item detection
TRAJECTORY_HISTORY_LENGTH = 100  # Maximum frames to keep in track history
MIN_TRAJECTORY_LENGTH = 100     # Minimum trajectory length in pixels to consider as moved (not used - relative comparison instead)
MIN_MOVEMENT_THRESHOLD = 20     # Minimum movement distance in pixels to consider as active (not used - relative comparison instead)
GRABBED_ITEM_DISPLAY = True     # Whether to display grabbed item information
TRAJECTORY_VISUALIZATION = True # Whether to visualize trajectories

# Output settings
SAVE_VIDEO = True
SAVE_STATS = False  # No statistics needed
STATS_FORMAT = 'json'  # 'json' or 'csv' 
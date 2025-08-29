"""
Application Configuration and Component Setup
"""

from pathlib import Path
from config import *
from models.detector import HygieneDetector
from tracking.tracker import HygieneTracker
from utils.video_io import VideoProcessor
from utils.video_annotation import VideoAnnotator


class AppConfigurator:
    """Handle application configuration and component initialization"""
    
    def validate_paths(self):
        """
        Validate required file paths exist
        
        Returns:
            bool: True if all paths are valid
        """
        # Check if model exists
        if not MODEL_PATH.exists():
            print(f" Model not found at: {MODEL_PATH}")
            print("Please ensure your trained model is in the correct location")
            return False
        
        # Check if video exists
        if not VIDEO_PATH.exists():
            print(f" Video not found at: {VIDEO_PATH}")
            print("Please ensure your video file is in the correct location")
            return False
        
        return True
    
    def setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        OUTPUT_PATH.mkdir(exist_ok=True)
    
    def initialize_components(self):
        """
        Initialize all system components
        
        Returns:
            tuple: (detector, tracker, processor, annotator)
        """
        print("\n Initializing components...")
        
        # Initialize detector
        detector = HygieneDetector(
            model_path=str(MODEL_PATH),
            confidence=MODEL_CONFIDENCE,
            iou_threshold=MODEL_IOU_THRESHOLD
        )
        
        # Initialize tracker
        tracker = HygieneTracker(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            track_lost_delay=TRACK_LOST_DELAY,
            track_frame_rate=TRACK_FRAME_RATE,
            detector=detector
        )
        
        # Initialize video processor
        output_video_path = OUTPUT_PATH / "output_video.mp4"
        processor = VideoProcessor(
            video_path=str(VIDEO_PATH),
            output_path=str(output_video_path)
        )
        
        # Initialize video annotator
        annotator = VideoAnnotator()
        
        # Setup video writer
        processor.setup_video_writer(
            output_path=str(output_video_path),
            fps=OUTPUT_FPS
        )
        
        return detector, tracker, processor, annotator
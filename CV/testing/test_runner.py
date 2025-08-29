"""
Testing Utilities
"""

from config import *
from models.detector import HygieneDetector
from tracking.tracker import HygieneTracker
from utils.video_io import VideoProcessor


class TestRunner:
    """Handle testing functionality"""
    
    def run_single_frame_test(self):
        """Test detection and tracking on a single frame"""
        print(" Running single frame test...")
        
        # Initialize components
        detector = HygieneDetector(str(MODEL_PATH))
        tracker = HygieneTracker(detector=detector)
        
        # Load video and get first frame
        processor = VideoProcessor(str(VIDEO_PATH))
        frame_gen = processor.get_frame_generator()
        _, frame = next(frame_gen)
        
        print(f" Testing with frame: {frame.shape}")
        
        # Test detection
        frame, detections = detector.detect_frame(frame)
        print(f" Detections: {len(detections)} objects")
        
        # Test tracking
        detections = tracker.update_tracks(detections, frame)
        print(f" Tracking updated: {len(detections)} tracks")
        
        print(" Single frame test completed")
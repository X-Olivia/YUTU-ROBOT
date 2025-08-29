"""
Core Pipeline for Detection and Tracking
"""

import time
from pathlib import Path
from config import *
from models.detector import HygieneDetector
from tracking.tracker import HygieneTracker
from utils.video_io import VideoProcessor
from utils.video_annotation import VideoAnnotator
from utils.progress_monitor import ProgressMonitor


class DetectionPipeline:
    """Main pipeline for detection and tracking processing"""
    
    def __init__(self, detector, tracker, processor, annotator):
        """
        Initialize pipeline with configured components
        
        Args:
            detector: HygieneDetector instance
            tracker: HygieneTracker instance  
            processor: VideoProcessor instance
            annotator: VideoAnnotator instance
        """
        self.detector = detector
        self.tracker = tracker
        self.processor = processor
        self.annotator = annotator
        self.progress_monitor = ProgressMonitor()
        
    def run(self):
        """
        Run the main detection and tracking pipeline
        
        Returns:
            dict: Processing results and statistics
        """
        print("\n Starting video processing...")
        start_time = time.time()
        
        # Process video frames
        frame_count = 0
        processed_frames = 0
        
        for frame_num, frame in self.processor.get_frame_generator(frame_skip=FRAME_SKIP):
            frame_count += 1
            
            # Detection and tracking pipeline
            frame, detections = self.detector.detect_frame(frame)
            detections = self.tracker.update_tracks(detections, frame)
            frame = self.annotator.annotate_frame(frame, detections, self.detector, self.tracker)
            
            # Write processed frame
            self.processor.write_frame(frame)
            processed_frames += 1
            
            # Update progress
            if frame_count % 100 == 0:
                self.progress_monitor.update_progress(
                    frame_count, processed_frames, start_time, 
                    self.tracker, self.detector
                )
        
        # Cleanup
        self.processor.close_writer()
        
        # Calculate final results
        total_time = time.time() - start_time
        final_grabbed_item = self.tracker.get_grabbed_item_info()
        
        # Add item name to final_grabbed_item for display
        if final_grabbed_item:
            final_grabbed_item['item_name'] = self.detector.get_class_names().get(
                final_grabbed_item['class_id'], 'Unknown'
            )
        
        return {
            'total_time': total_time,
            'processed_frames': processed_frames,
            'final_grabbed_item': final_grabbed_item
        }
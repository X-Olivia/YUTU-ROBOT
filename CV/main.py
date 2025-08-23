#!/usr/bin/env python3
"""
Main script for Detection and Tracking
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from models.detector import HygieneDetector
from tracking.tracker import HygieneTracker
from utils.video_processor import VideoProcessor, VideoAnnotator

def main():
    """Main function for detection and tracking"""
    print(" Detection and Tracking System")
    print("=" * 50)
    
    # Check if model and video exist
    if not MODEL_PATH.exists():
        print(f" Model not found at: {MODEL_PATH}")
        print("Please ensure your trained model is in the correct location")
        return
    
    if not VIDEO_PATH.exists():
        print(f" Video not found at: {VIDEO_PATH}")
        print("Please ensure your video file is in the correct location")
        return
    
    # Create output directory
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    try:
        # Initialize components
        print("\n Initializing components...")
        
        # Hygiene products detector
        detector = HygieneDetector(
            model_path=str(MODEL_PATH),
            confidence=MODEL_CONFIDENCE,
            iou_threshold=MODEL_IOU_THRESHOLD
        )
        
        # Hygiene products tracker
        tracker = HygieneTracker(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            track_lost_delay=TRACK_LOST_DELAY,
            track_frame_rate=TRACK_FRAME_RATE
        )
        
        # No counting line needed
        
        # Video processor
        output_video_path = OUTPUT_PATH / "output_video.mp4"
        processor = VideoProcessor(
            video_path=str(VIDEO_PATH),
            output_path=str(output_video_path)
        )
        
        # Video annotator
        annotator = VideoAnnotator()
        
        # Setup video writer
        writer = processor.setup_video_writer(
            output_path=str(output_video_path),
            fps=OUTPUT_FPS
        )
        
        print("\n Starting video processing...")
        start_time = time.time()
        
        # Process video frames
        frame_count = 0
        processed_frames = 0
        
        for frame_num, frame in processor.get_frame_generator(frame_skip=FRAME_SKIP):
            frame_count += 1
            
            # Hygiene products detection
            frame, detections = detector.detect_frame(frame)
            
            # Hygiene products tracking
            detections = tracker.update_tracks(detections, frame)
            
            # Annotate frame
            frame = annotator.annotate_frame(frame, detections, detector)
            
            # No counting line annotation needed
            
            # Write frame to output video
            processor.write_frame(frame)
            processed_frames += 1
            
            # Progress update
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed
                print(f" Processed {frame_count} frames, FPS: {fps:.2f}")
        
        # Cleanup
        processor.close_writer()
        
        # Final statistics
        total_time = time.time() - start_time
        
        print("\n Processing complete!")
        print("=" * 50)
        print(f" Input video: {VIDEO_PATH.name}")
        print(f" Output video: {output_video_path.name}")
        print(f" Total processing time: {total_time:.2f} seconds")
        print(f" Total frames processed: {processed_frames}")
        print(f" Output saved to: {OUTPUT_PATH}")
        print(f" Tracking information displayed in video")
        
    except KeyboardInterrupt:
        print("\n Processing interrupted by user")
        if 'processor' in locals():
            processor.close_writer()
    except Exception as e:
        print(f"\n Error during processing: {e}")
        if 'processor' in locals():
            processor.close_writer()
        raise

def run_single_frame_test():
    """Test detection and tracking on a single frame"""
    print(" Running single frame test...")
    
    # Initialize components
    detector = HygieneDetector(str(MODEL_PATH))
    tracker = HygieneTracker()
    
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

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_single_frame_test()
    else:
        main() 
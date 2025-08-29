"""
Progress Monitoring Utilities
"""

import time
from pathlib import Path
from config import *


class ProgressMonitor:
    """Monitor and display processing progress"""
    
    def update_progress(self, frame_count, processed_frames, start_time, tracker, detector):
        """
        Update and display processing progress
        
        Args:
            frame_count: Total frames processed
            processed_frames: Frames written to output
            start_time: Processing start timestamp
            tracker: HygieneTracker instance for grabbed item info
            detector: HygieneDetector instance for class names
        """
        elapsed = time.time() - start_time
        fps = processed_frames / elapsed
        
        # Get current grabbed item info
        grabbed_item = tracker.get_grabbed_item_info()
        if grabbed_item:
            item_info = f"Grabbed: {detector.get_class_names().get(grabbed_item['class_id'], 'Unknown')} #{grabbed_item['track_id']}"
            print(f" Processed {frame_count} frames, FPS: {fps:.2f} | {item_info}")
        else:
            print(f" Processed {frame_count} frames, FPS: {fps:.2f}")
    
    def display_final_results(self, results):
        """
        Display final processing results
        
        Args:
            results: dict containing processing results
        """
        print("\n Processing complete!")
        print("=" * 50)
        print(f" Input video: {VIDEO_PATH.name}")
        print(f" Output video: output_video.mp4")
        print(f" Total processing time: {results['total_time']:.2f} seconds")
        print(f" Total frames processed: {results['processed_frames']}")
        print(f" Output saved to: {OUTPUT_PATH}")
        print(f" Tracking information displayed in video")
        
        # Display final grabbed item information
        final_grabbed_item = results['final_grabbed_item']
        if final_grabbed_item:
            print("\n Grabbed Item Analysis:")
            print("-" * 30)
            print(f" Item: {final_grabbed_item.get('item_name', 'Unknown')}")
            print(f" Track ID: #{final_grabbed_item['track_id']}")
            print(f" Trajectory Length: {final_grabbed_item['trajectory_length']:.1f} pixels")
            print(f" Duration: {final_grabbed_item['duration_frames']} frames")
            print(f" Frame Range: {final_grabbed_item['start_frame']} - {final_grabbed_item['end_frame']}")
            
            # Display item matching results if available
            if 'match_result' in results and results['match_result']:
                print("\n Item Matching:")
                print("-" * 30)
                print(f" Matched Item: {results['match_result']['matched_item_name']}")
                print(f" Similarity: {results['match_result']['similarity']:.1%}")
        else:
            print("\n No significant item movement detected")
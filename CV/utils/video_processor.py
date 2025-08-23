"""
Video Processing Utilities
"""

import cv2
import numpy as np
import supervision as sv
from typing import Generator, Tuple, Optional
from pathlib import Path
import time

class VideoProcessor:
    """Video processing utilities for hygiene products detection and tracking"""
    
    def __init__(self, video_path: str, output_path: str = None):
        """
        Initialize video processor
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else None
        
        # Video info
        self.video_info = None
        self._load_video_info()
        
        # Video capture and writer
        self.cap = None
        self.writer = None
        
    def _load_video_info(self):
        """Load video information"""
        try:
            self.video_info = sv.VideoInfo.from_video_path(str(self.video_path))
            print(f" Video loaded: {self.video_info.width}x{self.video_info.height}, "
                  f"{self.video_info.fps:.2f} FPS, {self.video_info.total_frames} frames")
        except Exception as e:
            print(f" Error loading video: {e}")
            raise
    
    def get_frame_generator(self, frame_skip: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from video
        
        Args:
            frame_skip: Process every Nth frame
            
        Yields:
            Tuple of (frame_number, frame)
        """
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                yield frame_count, frame
                
            frame_count += 1
        
        self.cap.release()
    
    def setup_video_writer(self, output_path: str, fps: int = 30, 
                          width: int = None, height: int = None) -> cv2.VideoWriter:
        """
        Setup video writer for output
        
        Args:
            output_path: Output video path
            fps: Output frame rate
            width: Output width (uses input width if None)
            height: Output height (uses input height if None)
            
        Returns:
            VideoWriter instance
        """
        if width is None:
            width = self.video_info.width
        if height is None:
            height = self.video_info.height
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )
        
        print(f"Video writer setup: {width}x{height} @ {fps} FPS")
        return self.writer
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to output video"""
        if self.writer is not None:
            self.writer.write(frame)
    
    def close_writer(self):
        """Close video writer"""
        if self.writer is not None:
            self.writer.release()
            print("Video writer closed")
    
    def get_video_dimensions(self) -> Tuple[int, int]:
        """Get video dimensions"""
        return self.video_info.width, self.video_info.height
    
    def get_video_fps(self) -> float:
        """Get video frame rate"""
        return self.video_info.fps
    
    def get_total_frames(self) -> int:
        """Get total number of frames"""
        return self.video_info.total_frames

class VideoAnnotator:
    """Video annotation utilities"""
    
    def __init__(self):
        """Initialize video annotator"""
        self.box_annotator = sv.BoxAnnotator(thickness=4)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
        self.trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
        # No counting line annotator needed
    
    def annotate_frame(self, 
                      frame: np.ndarray, 
                      detections: sv.Detections,
                      detector: Optional[object] = None,
                      tracker: Optional[object] = None) -> np.ndarray:
        """
        Annotate frame with detections and tracking info
        
        Args:
            frame: Input frame
            detections: Detection results
            detector: Detector instance for class names
            tracker: Tracker instance for track history
            
        Returns:
            Annotated frame
        """
        # Annotate frame
        if len(detections) > 0:
            # Create labels using zip 
            labels = [
                f"#{tracker_id} {detector.get_class_names().get(class_id, 'Unknown') if detector else f'Class_{class_id}'} {confidence:0.2f}"
                for confidence, class_id, tracker_id
                in zip(detections.confidence, detections.class_id, detections.tracker_id)
                if tracker_id is not None and class_id is not None and confidence is not None
            ]
            
            # Use original notebook annotation order
            annotated_frame = frame.copy()
            annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # Add grabbed item visualization if tracker is provided
            if tracker is not None:
                annotated_frame = self._add_grabbed_item_visualization(annotated_frame, tracker, detector)
            
            return annotated_frame
        
        return frame
    
    def _add_grabbed_item_visualization(self, frame: np.ndarray, tracker: object, detector: object) -> np.ndarray:
        """Add visualization for grabbed item detection"""
        try:
            # Get grabbed item info
            grabbed_item = tracker.get_grabbed_item_info()
            if grabbed_item is None:
                return frame
            
            # Highlight the grabbed item with special color
            track_id = grabbed_item['track_id']
            class_id = grabbed_item['class_id']
            trajectory_length = grabbed_item['trajectory_length']
            
            # Find the detection for this track
            if hasattr(tracker, 'track_history') and track_id in tracker.track_history:
                # Draw trajectory for grabbed item
                track_history = tracker.get_track_history(track_id)
                if len(track_history) > 1:
                    # Convert points to numpy array for drawing
                    points = np.array(track_history, dtype=np.int32)
                    # Draw trajectory with special color (red for grabbed item)
                    cv2.polylines(frame, [points], False, (0, 0, 255), 3)
                
                # Add text overlay for grabbed item info
                class_name = detector.get_class_names().get(class_id, 'Unknown') if detector else f'Class_{class_id}'
                info_text = f"Grabbed: {class_name} #{track_id}"
                length_text = f"Trajectory: {trajectory_length:.1f}px"
                
                # Position text at top-left corner
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 255), 2)
                cv2.putText(frame, length_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 255), 2)
                
                # Add duration info if available
                if 'duration_frames' in grabbed_item:
                    duration_text = f"Duration: {grabbed_item['duration_frames']} frames"
                    cv2.putText(frame, duration_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 0, 255), 2)
        
        except Exception as e:
            # If visualization fails, just return the original frame
            print(f"Warning: Failed to add grabbed item visualization: {e}")
        
        return frame
    
    def _draw_track_history(self, frame: np.ndarray, detections: sv.Detections, 
                           tracker: object) -> np.ndarray:
        """Draw track history on frame"""
        if len(detections) == 0:
            return frame
        
        # Draw track history for each detection
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is not None:
                track_history = tracker.get_track_history(track_id)
                if len(track_history) > 1:
                    # Convert points to numpy array for drawing
                    points = np.array(track_history, dtype=np.int32)
                    cv2.polylines(frame, [points], False, (0, 255, 0), 2)
        
        return frame
    
    def draw_counting_line(self, frame: np.ndarray, 
                          start_point: Tuple[int, int], 
                          end_point: Tuple[int, int],
                          color: Tuple[int, int, int] = (255, 0, 0),
                          thickness: int = 3) -> np.ndarray:
        """Draw counting line on frame"""
        cv2.line(frame, start_point, end_point, color, thickness)
        return frame
    
    def add_text_overlay(self, frame: np.ndarray, text: str, 
                        position: Tuple[int, int] = (10, 30),
                        color: Tuple[int, int, int] = (255, 255, 255),
                        scale: float = 1.0,
                        thickness: int = 2) -> np.ndarray:
        """Add text overlay to frame"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   scale, color, thickness)
        return frame 
"""
Video Annotation Utilities
"""

import cv2
import numpy as np
import supervision as sv
from typing import Tuple, Optional

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
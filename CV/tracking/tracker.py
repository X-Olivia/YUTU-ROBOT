"""
Tracking Module using ByteTrack
"""

import numpy as np
import supervision as sv
from typing import List, Tuple, Dict, Optional
from config import TRAJECTORY_HISTORY_LENGTH, MIN_TRAJECTORY_LENGTH, MIN_MOVEMENT_THRESHOLD



class HygieneTracker:
    """Hygiene products tracking using ByteTrack"""
    
    def __init__(self, 
                 track_activation_threshold: float = 0.25,
                 track_lost_delay: int = 30,
                 track_frame_rate: int = 30,
                 detector: object = None):
        """
        Initialize tracker
        
        Args:
            track_activation_threshold: Threshold for track activation
            track_lost_delay: Frames to wait before marking track as lost
            track_frame_rate: Frame rate for tracking
            detector: Detector instance for class names
        """
        self.track_activation_threshold = track_activation_threshold
        self.track_lost_delay = track_lost_delay
        self.track_frame_rate = track_frame_rate
        self.detector = detector
        
        # Initialize ByteTracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=track_lost_delay,
            frame_rate=track_frame_rate
        )
        
        # Trajectory analysis data structures
        self.track_history = {}  # {track_id: [(frame_num, x, y, class_id), ...]}
        self.trajectory_lengths = {}  # {track_id: float}
        self.grabbed_item_info = None  # Current grabbed item information
        self.frame_count = 0
        
    def update_tracks(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """
        Update tracks
        
        Args:
            detections: Current frame detections
            frame: Current frame
            
        Returns:
            Updated detections with tracking IDs
        """
        # Update tracks
        detections = self.tracker.update_with_detections(detections)
        
        # Update trajectory analysis
        self._update_track_history(detections)
        self._identify_grabbed_item()
        
        return detections
    
    def _update_track_history(self, detections: sv.Detections):
        """Update track history for trajectory analysis"""
        self.frame_count += 1
        
        if len(detections) == 0:
            return
        
        # Update history for each track
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            if track_id is not None:
                # Get bounding box center
                bbox = detections.xyxy[i]
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
                class_id = detections.class_id[i]
                
                # Initialize track history if not exists
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                # Add current position to history
                self.track_history[track_id].append((self.frame_count, x, y, class_id))
                
                # Limit history length
                if len(self.track_history[track_id]) > TRAJECTORY_HISTORY_LENGTH:
                    self.track_history[track_id].pop(0)
                
                # Calculate trajectory length
                self._calculate_trajectory_length(track_id)
        
    
    def _calculate_trajectory_length(self, track_id: int):
        """Calculate trajectory length for a specific track"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            self.trajectory_lengths[track_id] = 0.0
            return
        
        history = self.track_history[track_id]
        total_length = 0.0
        
        # Calculate distance between consecutive positions
        for i in range(1, len(history)):
            prev_x, prev_y = history[i-1][1], history[i-1][2]
            curr_x, curr_y = history[i][1], history[i][2]
            
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_length += distance
        
        self.trajectory_lengths[track_id] = total_length
    
    
    def _identify_grabbed_item(self):
        """Identify the most likely grabbed item based on trajectory length"""
        if not self.trajectory_lengths:
            self.grabbed_item_info = None
            return
        
        # Filter out hands and find item with longest trajectory
        candidate_items = []
        
        for track_id, trajectory_length in self.trajectory_lengths.items():
            if track_id in self.track_history:
                # Get the most recent class_id for this track
                if self.track_history[track_id]:
                    class_id = self.track_history[track_id][-1][3]
                    class_name = self.get_class_names().get(class_id, f'Unknown_{class_id}')
                    
                    # Skip hands by class name instead of class_id
                    if class_name == 'hand':
                        continue
                    
                    # Include all items, no minimum threshold - we'll compare lengths
                    candidate_items.append((track_id, trajectory_length, class_id))
        
        # Select item with longest trajectory
        if candidate_items:
            # Sort by trajectory length (descending)
            candidate_items.sort(key=lambda x: x[1], reverse=True)
            best_track_id, best_length, best_class_id = candidate_items[0]
            
            # Get additional info
            if best_track_id in self.track_history:
                history = self.track_history[best_track_id]
                start_frame = history[0][0] if history else self.frame_count
                end_frame = history[-1][0] if history else self.frame_count
                
                self.grabbed_item_info = {
                    'track_id': best_track_id,
                    'class_id': best_class_id,
                    'trajectory_length': best_length,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'duration_frames': end_frame - start_frame + 1
                }
        else:
            self.grabbed_item_info = None
    
    def get_grabbed_item_info(self) -> Optional[Dict]:
        """Get information about the currently identified grabbed item"""
        return self.grabbed_item_info
    
    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Get position history for a specific track (for visualization)"""
        if track_id in self.track_history:
            # Return only x, y coordinates for visualization
            return [(int(pos[1]), int(pos[2])) for pos in self.track_history[track_id]]
        return []
    
    def get_trajectory_length(self, track_id: int) -> float:
        """Get trajectory length for a specific track"""
        return self.trajectory_lengths.get(track_id, 0.0)
    
    def get_class_names(self) -> Dict[int, str]:
        """Get class names dictionary from detector"""
        if self.detector:
            return self.detector.get_class_names()
        return {}
    

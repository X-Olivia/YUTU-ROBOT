"""
Frame Extraction Utilities for Item Matching
"""

import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path
from config import EXPORT_MATCHING_FRAMES, MATCHING_FRAMES_OUTPUT_DIR


class FrameExtractor:
    """Extract specific frames and detection boxes for item matching"""
    
    def __init__(self, video_path: str):
        """
        Initialize frame extractor
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
    
    def extract_grabbed_item_images(self, grabbed_item_info: dict, max_frames: int = 5) -> List[np.ndarray]:
        """
        Extract images of the grabbed item from video
        
        Args:
            grabbed_item_info: Information about the grabbed item
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of cropped images of the grabbed item
        """
        if not grabbed_item_info or 'track_boxes' not in grabbed_item_info:
            return []
        
        track_boxes = grabbed_item_info['track_boxes']
        if not track_boxes:
            return []
        
        # Filter frames by target class_id
        target_class_id = grabbed_item_info.get('class_id')
        if target_class_id is not None:
            # Filter to only frames with the target class_id
            filtered_boxes = [box for box in track_boxes if len(box) > 3 and box[3] == target_class_id]
            if not filtered_boxes:
                print(f"Warning: No frames found for target class_id {target_class_id}")
                return []
        else:
            filtered_boxes = track_boxes
        
        # Select best frames from filtered boxes
        selected_frames = self._select_best_frames(filtered_boxes, max_frames)
        
        # Extract images
        images = []
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Create export directory if needed
        if EXPORT_MATCHING_FRAMES:
            MATCHING_FRAMES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            track_id = grabbed_item_info.get('track_id', 'unknown')
        
        try:
            for i, frame_data in enumerate(selected_frames):
                # Handle both old format (frame_num, bbox, confidence) and new format (frame_num, bbox, confidence, class_id)
                frame_num, bbox, confidence = frame_data[0], frame_data[1], frame_data[2]
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # frame_num is 1-based
                ret, frame = cap.read()
                
                if ret:
                    # Crop detection box
                    cropped = self._crop_detection_box(frame, bbox)
                    if cropped is not None:
                        images.append(cropped)
                        
                        # Export frame if enabled
                        if EXPORT_MATCHING_FRAMES:
                            class_info = f"_class_{target_class_id}" if target_class_id is not None else ""
                            filename = f"track_{track_id}_frame_{frame_num}_conf_{confidence:.2f}{class_info}_{i+1}.jpg"
                            output_path = MATCHING_FRAMES_OUTPUT_DIR / filename
                            cv2.imwrite(str(output_path), cropped)
        finally:
            cap.release()
        
        if EXPORT_MATCHING_FRAMES and images:
            print(f"Exported {len(images)} matching frames to {MATCHING_FRAMES_OUTPUT_DIR}")
        
        return images
    
    def _select_best_frames(self, track_boxes: List[Tuple], max_frames: int) -> List[Tuple]:
        """
        Select the best frames for matching
        
        Args:
            track_boxes: List of (frame_num, bbox, confidence) or (frame_num, bbox, confidence, class_id) tuples
            max_frames: Maximum number of frames to select
            
        Returns:
            Selected frames sorted by quality
        """
        if len(track_boxes) <= max_frames:
            return track_boxes
        
        # Sort by confidence (descending) and bbox area (descending)
        def frame_quality(item):
            frame_num, bbox, confidence = item[0], item[1], item[2]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            return (confidence, area)
        
        sorted_frames = sorted(track_boxes, key=frame_quality, reverse=True)
        return sorted_frames[:max_frames]
    
    def _crop_detection_box(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Crop detection box from frame
        
        Args:
            frame: Full frame image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Cropped image or None if invalid
        """
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Crop the image
            cropped = frame[y1:y2, x1:x2]
            
            # Return None if crop is too small
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return None
                
            return cropped
            
        except Exception:
            return None
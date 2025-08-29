"""
Video I/O Utilities
"""

import cv2
import numpy as np
import supervision as sv
from typing import Generator, Tuple
from pathlib import Path

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
        
        print(f" Video writer setup: {width}x{height} @ {fps} FPS")
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
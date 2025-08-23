"""
Detection Module using YOLOv8
"""

import os
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import supervision as sv
from config import HYGIENE_CLASSES

class HygieneDetector:
    """Hygiene products detection using YOLOv8 model"""
    
    def __init__(self, model_path: str, confidence: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            print(f" Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.class_names = self.model.model.names
            print(f" Model loaded successfully! Classes: {len(self.class_names)}")
        except Exception as e:
            print(f" Error loading model: {e}")
            raise
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, sv.Detections]:
        """
        Detect hygiene in a single frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Run inference
        results = self.model(frame, conf=self.confidence, iou=self.iou_threshold, verbose=False)
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter for hygiene product classes only
        hygiene_detections = self._filter_hygiene_detections(detections)
        
        return frame, hygiene_detections
    
    def _filter_hygiene_detections(self, detections: sv.Detections) -> sv.Detections:
        """Filter detections to only include hygiene products and hands"""
        if len(detections) == 0:
            return detections
        
        # Get class names for detected objects
        class_names = [self.class_names[class_id] for class_id in detections.class_id]
        
        # Remove 'background' from detection classes
        hygiene_classes = [cls for cls in HYGIENE_CLASSES if cls != 'background']
        hygiene_mask = [name in hygiene_classes for name in class_names]
        
        # Apply mask to filter detections
        filtered_detections = detections[hygiene_mask]
        
        return filtered_detections
    
    def get_class_names(self) -> Dict[int, str]:
        """Get class names dictionary"""
        return self.class_names
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence,
            'iou_threshold': self.iou_threshold,
            'num_classes': len(self.class_names),
            'classes': self.class_names
        } 
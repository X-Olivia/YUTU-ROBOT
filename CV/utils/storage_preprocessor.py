"""
Storage Items Preprocessing Utilities
"""

import cv2
import numpy as np
from pathlib import Path
from models.detector import HygieneDetector
from config import MODEL_PATH, MODEL_CONFIDENCE, MODEL_IOU_THRESHOLD


class StoragePreprocessor:
    """Preprocess storage items with detection and cropping"""
    
    def __init__(self, storage_dir: str, output_dir: str):
        """
        Initialize storage preprocessor
        
        Args:
            storage_dir: Directory containing original storage images
            output_dir: Directory to save processed images
        """
        self.storage_dir = Path(storage_dir)
        self.output_dir = Path(output_dir)
        self.detector = None
    
    def process_storage_items(self):
        """Process all storage items with detection and cropping"""
        if not self.storage_dir.exists():
            print(f"Storage directory {self.storage_dir} does not exist")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self.detector = HygieneDetector(
            model_path=str(MODEL_PATH),
            confidence=MODEL_CONFIDENCE,
            iou_threshold=MODEL_IOU_THRESHOLD
        )
        
        print(f"Processing storage items from {self.storage_dir}")
        
        processed_count = 0
        # Process jpg files
        for img_path in self.storage_dir.glob("*.jpg"):
            if self._process_single_image(img_path):
                processed_count += 1
        
        # Process JPG files
        for img_path in self.storage_dir.glob("*.JPG"):
            if self._process_single_image(img_path):
                processed_count += 1
        
        print(f"Processed {processed_count} storage images")
    
    def _process_single_image(self, img_path: Path) -> bool:
        """
        Process a single storage image
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if processed successfully
        """
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load {img_path}")
                return False
            
            # Detect items
            _, detections = self.detector.detect_frame(img)
            
            if len(detections) > 0:
                # Select best detection (highest confidence)
                best_idx = np.argmax(detections.confidence)
                best_bbox = detections.xyxy[best_idx]
                
                # Crop detection box
                cropped = self._crop_detection_box(img, best_bbox)
                
                if cropped is not None:
                    # Save cropped image
                    output_name = f"{img_path.stem}_cropped.jpg"
                    output_path = self.output_dir / output_name
                    cv2.imwrite(str(output_path), cropped)
                    print(f"Processed: {img_path.name} -> {output_name}")
                    return True
            
            print(f"No detection found in {img_path.name}")
            return False
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False
    
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
    
    def process_if_needed(self):
        """Process storage items if output directory doesn't exist or is empty"""
        if not self.output_dir.exists() or not any(self.output_dir.glob("*.jpg")):
            print("Storage items need preprocessing...")
            self.process_storage_items()
        else:
            print("Storage items already processed")
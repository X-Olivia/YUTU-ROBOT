"""
Color Feature Matching for Item Recognition
"""

import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path


class ColorMatcher:
    """Match items based on color features in Lab color space"""
    
    def __init__(self, storage_dir: str):
        """
        Initialize color matcher
        
        Args:
            storage_dir: Directory containing reference images
        """
        # Check if processed directory exists, use it if available
        processed_dir = Path(str(storage_dir) + "_processed")
        if processed_dir.exists() and any(processed_dir.glob("*.jpg")):
            self.storage_dir = processed_dir
            print(f"Using processed storage images from {processed_dir}")
        else:
            self.storage_dir = Path(storage_dir)
            print(f"Using original storage images from {storage_dir}")
        
        self.storage_images = {}
        self.storage_color_features = {}
        
        self._load_storage_images()
        self._precompute_color_features()
    
    def _load_storage_images(self):
        """Load all storage images"""
        if not self.storage_dir.exists():
            print(f"Warning: Storage directory {self.storage_dir} does not exist")
            return
        
        # Load jpg files
        for img_path in self.storage_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.storage_images[img_path.name] = img
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        # Load JPG files
        for img_path in self.storage_dir.glob("*.JPG"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.storage_images[img_path.name] = img
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        print(f"Loaded {len(self.storage_images)} storage images for color matching")
    
    def _precompute_color_features(self):
        """Precompute color features for all storage images"""
        print("Computing color features for storage images...")
        
        for name, img in self.storage_images.items():
            color_feature = self._extract_color_features(img)
            self.storage_color_features[name] = color_feature
        
        print(f"Computed color features for {len(self.storage_color_features)} storage images")
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color features from image in Lab color space
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Color feature vector
        """
        # Convert to Lab color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract histograms for each channel
        l_hist = cv2.calcHist([lab_image], [0], None, [32], [0, 256])
        a_hist = cv2.calcHist([lab_image], [1], None, [32], [0, 256])
        b_hist = cv2.calcHist([lab_image], [2], None, [32], [0, 256])
        
        # Normalize histograms
        l_hist = cv2.normalize(l_hist, None).flatten()
        a_hist = cv2.normalize(a_hist, None).flatten()
        b_hist = cv2.normalize(b_hist, None).flatten()
        
        # Concatenate features
        color_features = np.concatenate([l_hist, a_hist, b_hist])
        
        return color_features
    
    def match_with_storage(self, item_images: List[np.ndarray]) -> Dict:
        """
        Match item images with storage images using color features
        
        Args:
            item_images: List of cropped item images
            
        Returns:
            Match result dictionary
        """
        if not item_images or not self.storage_color_features:
            return {"matched_item_name": "No match", "similarity": 0.0}
        
        best_match = None
        best_score = 0.0
        
        # Extract color features from item images
        item_color_features = []
        for item_img in item_images:
            color_feature = self._extract_color_features(item_img)
            item_color_features.append(color_feature)
        
        # Average color features across all item images
        avg_item_feature = np.mean(item_color_features, axis=0)
        
        # Compare with storage images
        for storage_name, storage_feature in self.storage_color_features.items():
            # Calculate color similarity using correlation coefficient
            correlation = np.corrcoef(avg_item_feature, storage_feature)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            # Convert to similarity score (0-1)
            similarity = max(0.0, correlation)
            
            print(f"  {storage_name}: {similarity:.3f}")
            
            if similarity > best_score:
                best_score = similarity
                best_match = storage_name
        
        return {
            "matched_item_name": best_match or "No match",
            "similarity": best_score
        }
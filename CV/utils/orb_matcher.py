"""
ORB Feature Matching for Item Recognition
"""

import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path


class ORBMatcher:
    """Match items based on ORB features and RANSAC verification"""
    
    def __init__(self, storage_dir: str):
        """
        Initialize ORB matcher
        
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
        self._load_storage_images()
        
        # Initialize ORB detector
        from config import ORB_NFEATURES
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _load_storage_images(self):
        """Load all storage images"""
        if not self.storage_dir.exists():
            print(f"Warning: Storage directory {self.storage_dir} does not exist")
            return
        
        # Load all image files
        for img_path in self.storage_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.storage_images[img_path.name] = img
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        # Also try JPG extension
        for img_path in self.storage_dir.glob("*.JPG"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.storage_images[img_path.name] = img
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        print(f"Loaded {len(self.storage_images)} storage images")
    
    def match_with_storage(self, item_images: List[np.ndarray]) -> Dict:
        """
        Match item images with storage images using ORB features
        
        Args:
            item_images: List of cropped item images
            
        Returns:
            Match result dictionary
        """
        if not item_images or not self.storage_images:
            return {"matched_item_name": "No match", "similarity": 0.0}
        
        best_match = None
        best_score = 0.0
        
        # Compare with each storage image
        for storage_name, storage_img in self.storage_images.items():
            # Calculate similarity with all item images
            scores = []
            for item_img in item_images:
                score = self._orb_match_single(item_img, storage_img)
                scores.append(score)
            
            # Use average score
            avg_score = np.mean(scores)
            print(f"  {storage_name}: {avg_score:.3f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = storage_name
        
        return {
            "matched_item_name": best_match or "No match",
            "similarity": best_score
        }
    
    def _orb_match_single(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Match two images using ORB features with improved scoring
        
        Args:
            img1: First image (video frame)
            img2: Second image (storage image)
            
        Returns:
            Match score (0-1)
        """
        # Resize storage image to be closer to video frame size for better matching
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Scale down the larger image to make sizes more comparable
        max_size = 800
        if max(h2, w2) > max_size:
            scale = max_size / max(h2, w2)
            new_w2 = int(w2 * scale)
            new_h2 = int(h2 * scale)
            img2_resized = cv2.resize(img2, (new_w2, new_h2))
        else:
            img2_resized = img2
        
        # Extract ORB features
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2_resized, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # Simple match with distance threshold (for low-feature images)
        matches = self.matcher.match(des1, des2)
        
        # Sort by distance and take best matches
        matches = sorted(matches, key=lambda x: x.distance)
        
        from config import MIN_MATCH_COUNT
        if len(matches) < MIN_MATCH_COUNT:
            return 0.0
        
        # Use top 50% of matches or at least MIN_MATCH_COUNT
        good_match_count = max(MIN_MATCH_COUNT, len(matches) // 2)
        good_matches = matches[:good_match_count]
        
        # Skip RANSAC for very few matches - just use match quality
        if len(good_matches) < 8:
            # Simple scoring based on match distances
            avg_distance = np.mean([m.distance for m in good_matches])
            max_distance = 100.0  # Typical ORB distance range
            score = max(0, (max_distance - avg_distance) / max_distance)
            return score
        
        # RANSAC verification for sufficient matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        from config import RANSAC_THRESHOLD
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
        
        if mask is None:
            # Fallback to distance-based scoring
            avg_distance = np.mean([m.distance for m in good_matches])
            score = max(0, (100.0 - avg_distance) / 100.0)
            return score
        
        # Calculate score based on inliers
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)
        
        return inlier_ratio
"""
Packaging Material Classification Utilities
"""

import cv2
import numpy as np


class MaterialClassifier:
    """Classify packaging material: rigid box vs soft pouch"""
    
    def classify(self, image: np.ndarray) -> str:
        """
        Classify packaging material type
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            'rigid_box' or 'soft_pouch'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corner_count = len(corners) if corners is not None else 0
        corner_density = corner_count / (gray.shape[0] * gray.shape[1] / 10000)  # per 100x100 area
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_regularity = self._analyze_contour_regularity(contours)
        
        # Refined decision logic based on feature analysis
        # Rigid boxes: moderate corner density + reasonable contour regularity
        # Soft pouches: either very high corner density OR low contour regularity
        if (edge_density > 0.002 and 
            corner_density > 0.25 and corner_density < 1.0 and 
            contour_regularity > 0.15 and contour_regularity < 0.4):
            return "rigid_box"
        else:
            return "soft_pouch"
    
    def _analyze_contour_regularity(self, contours) -> float:
        """Analyze how regular/rectangular the main contours are"""
        if not contours:
            return 0.0
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Calculate regularity score based on polygon approximation
        if len(approx) == 4:
            # Rectangle-like shape (good for rigid boxes)
            return 0.8
        elif len(approx) <= 6:
            # Somewhat regular
            return 0.5
        else:
            # Irregular shape (typical for soft pouches)
            return 0.2
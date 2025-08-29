"""
CLIP Feature Matching for Item Recognition
"""

import cv2
import torch
import open_clip
import numpy as np
from typing import List, Dict
from pathlib import Path
from PIL import Image


class CLIPMatcher:
    """Match items based on CLIP semantic features"""
    
    def __init__(self, storage_dir: str):
        """
        Initialize CLIP matcher
        
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
        self.storage_features = {}
        
        # Initialize CLIP model with correct configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use laion2b_s34b_b79k which has consistent config, or specify quick_gelu explicitly
        try:
            # Try with explicit quick_gelu configuration to match OpenAI weights
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress the warning we're fixing
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', 
                    pretrained='openai',
                    force_quick_gelu=True  # Force QuickGELU to match OpenAI weights
                )
        except TypeError:
            # Fallback to alternative pretrained weights without config mismatch
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained='laion2b_s34b_b79k'  # Alternative pretrained weights
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._load_storage_images()
        self._precompute_storage_features()
    
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
    
    def _precompute_storage_features(self):
        """Precompute CLIP features for all storage images"""
        print("Computing CLIP features for storage images...")
        
        with torch.no_grad():
            for name, img in self.storage_images.items():
                # Convert BGR to RGB and then to PIL
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # Preprocess and encode
                image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                self.storage_features[name] = image_features.cpu()
        
        print(f"Computed features for {len(self.storage_features)} storage images")
    
    def match_with_storage(self, item_images: List[np.ndarray]) -> Dict:
        """
        Match item images with storage images using CLIP features
        
        Args:
            item_images: List of cropped item images
            
        Returns:
            Match result dictionary
        """
        if not item_images or not self.storage_features:
            return {"matched_item_name": "No match", "similarity": 0.0}
        
        best_match = None
        best_score = 0.0
        
        with torch.no_grad():
            # Extract features from all item images
            item_features = []
            for item_img in item_images:
                feature = self._extract_clip_features(item_img)
                if feature is not None:
                    item_features.append(feature)
            
            if not item_features:
                return {"matched_item_name": "No match", "similarity": 0.0}
            
            # Compare with each storage image
            for storage_name, storage_feature in self.storage_features.items():
                # Calculate similarities with all item images
                similarities = []
                for item_feature in item_features:
                    # Cosine similarity
                    similarity = torch.cosine_similarity(item_feature, storage_feature).item()
                    similarities.append(similarity)
                
                # Use average similarity
                avg_similarity = np.mean(similarities)
                print(f"  {storage_name}: {avg_similarity:.3f}")
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_match = storage_name
        
        return {
            "matched_item_name": best_match or "No match",
            "similarity": best_score
        }
    
    def _extract_clip_features(self, image: np.ndarray):
        """
        Extract CLIP features from a single image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Normalized CLIP feature vector
        """
        try:
            # Convert BGR to RGB and then to PIL
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess and encode
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu()
            
        except Exception as e:
            print(f"Warning: Failed to extract CLIP features: {e}")
            return None
"""
Multi-Signal Fusion Matcher for Item Recognition
"""

import numpy as np
from typing import List, Dict
from utils.clip_matcher import CLIPMatcher
from utils.color_matcher import ColorMatcher
from utils.material_classifier import MaterialClassifier


class FusionMatcher:
    """
    Fusion matcher combining CLIP, Color, and Material features
    Replaces hybrid_matcher with multi-signal approach
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize fusion matcher
        
        Args:
            storage_dir: Directory containing reference images
        """
        self.storage_dir = storage_dir
        
        # Initialize all matchers
        print("Initializing fusion matcher components...")
        self.clip_matcher = CLIPMatcher(storage_dir)
        self.color_matcher = ColorMatcher(storage_dir)
        self.material_classifier = MaterialClassifier()
        
        # Precompute material types for storage images
        self.storage_materials = {}
        self._precompute_storage_materials()
        
        # Signal weights (sum to 1.0)
        self.weights = {
            'clip': 0.6,        # CLIP semantic features (primary)
            'color': 0.1,       # Color features (auxiliary)
            'material': 0.3     # Material consistency (secondary)
        }
        
        print("Fusion matcher initialized with signal weights:", self.weights)
    
    def _precompute_storage_materials(self):
        """Precompute material types for all storage images"""
        print("Computing material types for storage images...")
        
        for name, img in self.clip_matcher.storage_images.items():
            material_type = self.material_classifier.classify(img)
            self.storage_materials[name] = material_type
            print(f"  {name}: {material_type}")
    
    def match_with_storage(self, item_images: List[np.ndarray]) -> Dict:
        """
        Match item images using multi-signal fusion
        
        Args:
            item_images: List of cropped item images
            
        Returns:
            Match result dictionary with fusion details
        """
        if not item_images:
            return {"matched_item_name": "No match", "similarity": 0.0}
        
        print("\n=== Multi-Signal Fusion Matching ===")
        
        # Step 1: Compute all scores for all candidates (simplified approach)
        print("1. CLIP semantic matching...")
        self._compute_all_clip_scores(item_images)
        
        print("2. Color feature matching...")
        self._compute_all_color_scores(item_images)
        
        print("3. Material consistency matching...")
        self._compute_all_material_scores(item_images)
        
        # Step 2: Build complete candidate matrix with all scores
        candidates = self._build_complete_candidates()
        
        if not candidates:
            return {"matched_item_name": "No match", "similarity": 0.0}
        
        # Step 3: Fusion scoring
        print("4. Fusion scoring...")
        fusion_results = self._calculate_fusion_scores(candidates)
        
        # Step 4: Select best match
        best_result = max(fusion_results, key=lambda x: x['fusion_score'])
        
        # Step 5: Format final result
        final_result = {
            "matched_item_name": best_result['item_name'],
            "similarity": best_result['fusion_score'],
            "fusion_details": {
                "clip_score": best_result.get('clip', 0.0),
                "color_score": best_result.get('color', 0.0),
                "material_score": best_result.get('material', 0.0),
                "weights_used": self.weights,
                "method": "multi_signal_fusion"
            }
        }
        
        print(f"Best match: {final_result['matched_item_name']} "
              f"(fusion score: {final_result['similarity']:.3f})")
        print("Signal contributions:")
        for signal, score in best_result.items():
            if signal in self.weights:
                weighted_contrib = score * self.weights[signal]
                print(f"  {signal}: {score:.3f} × {self.weights[signal]} = {weighted_contrib:.3f}")
        
        return final_result
    
    def _collect_candidates(self, results: List[Dict]) -> Dict:
        """
        Collect all unique candidates from different matchers
        
        Args:
            results: List of results from different matchers
            
        Returns:
            Dictionary of candidates with their scores
        """
        candidates = {}
        
        for i, result in enumerate(results):
            item_name = result.get('matched_item_name', 'No match')
            similarity = result.get('similarity', 0.0)
            
            if item_name != 'No match' and similarity > 0:
                if item_name not in candidates:
                    candidates[item_name] = {}
                
                # Map index to signal name
                signal_names = ['clip', 'color', 'material']
                signal = signal_names[i]
                candidates[item_name][signal] = similarity
        
        return candidates
    
    def _compute_all_clip_scores(self, item_images: List[np.ndarray]):
        """Compute CLIP scores for all storage items"""
        self._clip_scores = {}
        
        # Extract features from all item images
        item_features = []
        for item_img in item_images:
            feature = self.clip_matcher._extract_clip_features(item_img)
            if feature is not None:
                item_features.append(feature)
        
        if not item_features:
            return
        
        # Compare with each storage image
        for storage_name, storage_feature in self.clip_matcher.storage_features.items():
            similarities = []
            for item_feature in item_features:
                import torch
                similarity = torch.cosine_similarity(item_feature, storage_feature).item()
                similarities.append(similarity)
            
            # Use average similarity
            avg_similarity = np.mean(similarities)
            self._clip_scores[storage_name] = avg_similarity
            print(f"  {storage_name}: {avg_similarity:.3f}")
    
    def _compute_all_color_scores(self, item_images: List[np.ndarray]):
        """Compute color scores for all storage items"""
        self._color_scores = {}
        
        # Extract color features from item images
        item_color_features = []
        for item_img in item_images:
            color_feature = self.color_matcher._extract_color_features(item_img)
            item_color_features.append(color_feature)
        
        # Average color features across all item images
        avg_item_feature = np.mean(item_color_features, axis=0)
        
        # Compare with storage images
        for storage_name, storage_feature in self.color_matcher.storage_color_features.items():
            # Calculate color similarity using correlation coefficient
            correlation = np.corrcoef(avg_item_feature, storage_feature)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            # Convert to similarity score (0-1)
            similarity = max(0.0, correlation)
            self._color_scores[storage_name] = similarity
            print(f"  {storage_name}: {similarity:.3f}")
    
    def _compute_all_material_scores(self, item_images: List[np.ndarray]):
        """Compute material scores for all storage items - same as before"""
        self._material_consistency_matching(item_images)
    
    def _build_complete_candidates(self) -> Dict:
        """Build complete candidate matrix with all signal scores"""
        candidates = {}
        
        # Get all storage item names
        all_items = set()
        if hasattr(self, '_clip_scores'):
            all_items.update(self._clip_scores.keys())
        if hasattr(self, '_color_scores'):
            all_items.update(self._color_scores.keys())
        if hasattr(self, '_material_scores'):
            all_items.update(self._material_scores.keys())
        
        # Build candidate matrix
        for item_name in all_items:
            candidates[item_name] = {
                'clip': self._clip_scores.get(item_name, 0.0),
                'color': self._color_scores.get(item_name, 0.0),
                'material': self._material_scores.get(item_name, 0.0)
            }
        
        return candidates
    
    def _material_consistency_matching(self, item_images: List[np.ndarray]) -> Dict:
        """
        Material consistency matching - returns scores for ALL storage items
        
        Args:
            item_images: List of cropped item images
            
        Returns:
            Match result dictionary with all material scores
        """
        if not item_images or not self.storage_materials:
            return {"matched_item_name": "No match", "similarity": 0.0}
        
        # Classify material type of video item using first frame
        video_material = self.material_classifier.classify(item_images[0])
        print(f"  Video item material: {video_material}")
        
        # Calculate material scores for ALL storage items
        material_scores = {}
        best_match = None
        best_score = 0.0
        
        # Compare with storage materials
        for storage_name, storage_material in self.storage_materials.items():
            if storage_material == video_material:
                # Material match - high score
                similarity = 1.0
                print(f"  Material match: {storage_name}")
            else:
                # Material mismatch - low score but not zero
                similarity = 0.2
                print(f"  Material mismatch: {storage_name} ({storage_material} vs {video_material})")
            
            material_scores[storage_name] = similarity
            
            if similarity > best_score:
                best_score = similarity
                best_match = storage_name
        
        # Store all material scores for fusion calculation
        self._material_scores = material_scores
        
        return {
            "matched_item_name": best_match or "No match",
            "similarity": best_score
        }
    
    def _calculate_fusion_scores(self, candidates: Dict) -> List[Dict]:
        """
        Calculate fusion scores for all candidates
        
        Args:
            candidates: Dictionary of candidates with their signal scores
            
        Returns:
            List of candidates with fusion scores
        """
        fusion_results = []
        
        for item_name, scores in candidates.items():
            # Calculate weighted fusion score
            fusion_score = 0.0
            
            for signal, weight in self.weights.items():
                signal_score = scores.get(signal, 0.0)
                fusion_score += signal_score * weight
            
            # Create result record
            result = {
                'item_name': item_name,
                'fusion_score': fusion_score,
                **scores  # Include individual signal scores
            }
            
            fusion_results.append(result)
        
        return fusion_results
    
    def adjust_weights(self, new_weights: Dict):
        """
        Adjust fusion weights
        
        Args:
            new_weights: New weight dictionary
        """
        # Normalize weights to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in new_weights.items()}
            print("Updated fusion weights:", self.weights)
        else:
            print("Warning: Invalid weights provided, keeping current weights")
    
    def get_signal_availability(self) -> Dict:
        """
        Check availability of different matching signals
        
        Returns:
            Dictionary indicating which signals are available
        """
        return {
            'clip': len(self.clip_matcher.storage_features) > 0,
            'color': len(self.color_matcher.storage_color_features) > 0,
            'material': len(self.storage_materials) > 0
        }
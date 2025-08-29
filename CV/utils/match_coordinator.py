"""
Match Coordinator for Item Recognition
"""

from typing import Dict, Optional
from utils.frame_extractor import FrameExtractor
from utils.fusion_matcher import FusionMatcher


class MatchCoordinator:
    """Coordinate the complete item matching process"""
    
    def __init__(self, storage_dir: str):
        """
        Initialize match coordinator
        
        Args:
            storage_dir: Directory containing reference images
        """
        self.storage_dir = storage_dir
        self.fusion_matcher = FusionMatcher(storage_dir)
    
    def run_item_matching(self, video_path: str, grabbed_item_info: Dict) -> Optional[Dict]:
        """
        Run the complete item matching process
        
        Args:
            video_path: Path to the video file
            grabbed_item_info: Information about the grabbed item
            
        Returns:
            Match result or None if no match
        """
        if not grabbed_item_info:
            return None
        
        print(f"Starting item matching for track #{grabbed_item_info['track_id']}...")
        
        # Step 1: Extract frames
        frame_extractor = FrameExtractor(video_path)
        item_images = frame_extractor.extract_grabbed_item_images(grabbed_item_info)
        
        if not item_images:
            print("No valid frames extracted for matching")
            return None
        
        print(f"Extracted {len(item_images)} frames for matching")
        
        # Step 2: Match with storage using multi-signal fusion
        match_result = self.fusion_matcher.match_with_storage(item_images)
        
        # Fusion similarity threshold (weighted score range: 0-1, typical good matches > 0.4)
        if match_result['similarity'] > 0.4:
            print(f"Match found: {match_result['matched_item_name']} (fusion score: {match_result['similarity']:.3f})")
            if 'fusion_details' in match_result:
                print("Fusion details:", match_result['fusion_details']['method'])
        else:
            print("No confident match found")
        
        return match_result
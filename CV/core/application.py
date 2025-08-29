"""
Application Controller for Detection and Tracking System
"""

from core.app_configurator import AppConfigurator
from core.pipeline import DetectionPipeline
from utils.progress_monitor import ProgressMonitor
from utils.match_coordinator import MatchCoordinator
from utils.storage_preprocessor import StoragePreprocessor
from config import ITEM_MATCHING_ENABLED, STORAGE_ITEMS_DIR, STORAGE_ITEMS_PROCESSED_DIR, STORAGE_PREPROCESSING_ENABLED, VIDEO_PATH


class Application:
    """Main application controller"""
    
    def __init__(self):
        """Initialize application"""
        self.configurator = AppConfigurator()
        self.progress_monitor = ProgressMonitor()
    
    def run(self):
        """Run the complete application"""
        print(" Detection and Tracking System")
        print("=" * 50)
        
        # Validate system requirements
        if not self.configurator.validate_paths():
            return
        
        self.configurator.setup_output_directory()
        
        # Preprocess storage items if enabled
        if STORAGE_PREPROCESSING_ENABLED:
            preprocessor = StoragePreprocessor(str(STORAGE_ITEMS_DIR), str(STORAGE_ITEMS_PROCESSED_DIR))
            preprocessor.process_if_needed()
        
        try:
            # Initialize components
            detector, tracker, processor, annotator = self.configurator.initialize_components()
            
            # Run detection and tracking pipeline
            pipeline = DetectionPipeline(detector, tracker, processor, annotator)
            results = pipeline.run()
            
            # Run item matching if enabled and grabbed item exists
            if ITEM_MATCHING_ENABLED and results['final_grabbed_item']:
                try:
                    coordinator = MatchCoordinator(str(STORAGE_ITEMS_DIR))
                    match_result = coordinator.run_item_matching(str(VIDEO_PATH), results['final_grabbed_item'])
                    results['match_result'] = match_result
                except Exception as e:
                    print(f" Item matching failed: {e}")
                    results['match_result'] = None
            
            # Display final results
            self.progress_monitor.display_final_results(results)
            
        except KeyboardInterrupt:
            print("\n Processing interrupted by user")
            if 'processor' in locals():
                processor.close_writer()
        except Exception as e:
            print(f"\n Error during processing: {e}")
            if 'processor' in locals():
                processor.close_writer()
            raise
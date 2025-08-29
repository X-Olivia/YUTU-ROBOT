#!/usr/bin/env python3
"""
Main script for Detection and Tracking
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.application import Application
from testing.test_runner import TestRunner

def main():
    """Main function for detection and tracking"""
    app = Application()
    app.run()

def run_single_frame_test():
    """Test detection and tracking on a single frame"""
    test_runner = TestRunner()
    test_runner.run_single_frame_test()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_single_frame_test()
    else:
        main() 
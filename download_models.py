#!/usr/bin/env python3
"""
Model Download Script

Downloads all required pre-trained models for the CCTV Analytics System.
Run this script before starting the main application.

Usage:
    python download_models.py
"""

import os
import urllib.request
from pathlib import Path

# Model URLs - using working sources
MODELS = {
    # Face Detection (SSD) - from OpenCV GitHub
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    
    # Age Model - from spmallick's learnopencv repo (widely used mirror)
    "age_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel",
    
    # Gender Model - from spmallick's learnopencv repo
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading: {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Download all models."""
    # Determine models directory
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CCTV Analytics - Model Downloader")
    print("=" * 60)
    print(f"Downloading to: {models_dir}")
    print()
    
    success_count = 0
    skip_count = 0
    
    for filename, url in MODELS.items():
        dest = models_dir / filename
        
        if dest.exists():
            print(f"Skipping (exists): {filename}")
            skip_count += 1
            continue
        
        if download_file(url, dest):
            success_count += 1
    
    print()
    print("=" * 60)
    print(f"Downloaded: {success_count}, Skipped: {skip_count}, Failed: {len(MODELS) - success_count - skip_count}")
    
    # Verify all files exist
    print()
    print("Verifying models...")
    all_present = True
    for filename in MODELS.keys():
        path = models_dir / filename
        status = "✓" if path.exists() else "✗ MISSING"
        if not path.exists():
            all_present = False
        print(f"  {status}: {filename}")
    
    if all_present:
        print()
        print("✓ All models ready! You can now run: python -m src.main")
    else:
        print()
        print("✗ Some models are missing. Please download manually.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the CCTV Analytics Web Dashboard

Usage:
    python run_dashboard.py
    
Then open http://localhost:5000 in your browser.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.web_server import run_server

if __name__ == '__main__':
    print("=" * 60)
    print("🎥 CCTV Analytics Dashboard")
    print("=" * 60)
    print()
    print("Opening dashboard at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    print()
    
    run_server(host='0.0.0.0', port=8080, camera_id='dashboard_cam')

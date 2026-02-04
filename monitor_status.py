#!/usr/bin/env python3
"""
Remote Monitoring Status Script

Check the health and status of the face detection system remotely.
Shows recent detections, system stats, and validates detection quality.

Usage:
    python monitor_status.py --log detections.jsonl
    python monitor_status.py --log detections.jsonl --snapshots snapshots/
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import os


def read_recent_detections(log_path: Path, minutes: int = 60) -> list:
    """Read detections from the last N minutes."""
    if not log_path.exists():
        return []

    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent = []

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Parse timestamp
                ts_str = entry.get('timestamp', '')
                if ts_str:
                    ts = datetime.strptime(ts_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff_time:
                        recent.append(entry)
            except (json.JSONDecodeError, ValueError):
                continue

    return recent


def analyze_detections(detections: list) -> dict:
    """Analyze detection quality and statistics."""
    if not detections:
        return {"status": "NO_DATA", "message": "No recent detections found"}

    total_entries = len(detections)
    total_persons = sum(len(d.get('detections', [])) for d in detections)

    # Gender distribution
    genders = []
    ages = []
    confidences = []
    person_ids = set()

    for entry in detections:
        for det in entry.get('detections', []):
            genders.append(det.get('gender', 'unknown'))
            if det.get('age', 0) > 0:
                ages.append(det['age'])
            if det.get('confidence', 0) > 0:
                confidences.append(det['confidence'])
            if det.get('id'):
                person_ids.add(det['id'])

    gender_counts = Counter(genders)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    avg_age = sum(ages) / len(ages) if ages else 0

    # Time range
    timestamps = []
    for entry in detections:
        ts_str = entry.get('timestamp', '')
        if ts_str:
            try:
                ts = datetime.strptime(ts_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                timestamps.append(ts)
            except ValueError:
                pass

    time_range = ""
    if timestamps:
        time_range = f"{min(timestamps).strftime('%H:%M:%S')} - {max(timestamps).strftime('%H:%M:%S')}"

    return {
        "status": "OK" if avg_confidence >= 0.5 else "LOW_CONFIDENCE",
        "log_entries": total_entries,
        "total_detections": total_persons,
        "unique_persons": len(person_ids),
        "gender_distribution": dict(gender_counts),
        "average_confidence": round(avg_confidence, 3),
        "average_age": round(avg_age, 1) if ages else "N/A",
        "time_range": time_range,
        "sensor_id": detections[0].get('sensor_id', 'unknown') if detections else 'unknown'
    }


def check_snapshots(snapshot_dir: Path, hours: int = 24) -> dict:
    """Check recent snapshots."""
    if not snapshot_dir or not snapshot_dir.exists():
        return {"status": "NO_SNAPSHOTS", "count": 0}

    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_snapshots = []

    for f in snapshot_dir.glob("*.jpg"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime >= cutoff_time:
            recent_snapshots.append({
                "name": f.name,
                "time": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(f.stat().st_size / 1024, 1)
            })

    recent_snapshots.sort(key=lambda x: x['time'], reverse=True)

    return {
        "status": "OK" if recent_snapshots else "NO_RECENT",
        "count": len(recent_snapshots),
        "latest": recent_snapshots[:5] if recent_snapshots else []
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor face detection system status")
    parser.add_argument('--log', type=str, required=True, help='Path to detections.jsonl')
    parser.add_argument('--snapshots', type=str, help='Path to snapshots directory')
    parser.add_argument('--minutes', type=int, default=60, help='Analyze last N minutes (default: 60)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    log_path = Path(args.log)
    snapshot_dir = Path(args.snapshots) if args.snapshots else None

    # Analyze detections
    recent_detections = read_recent_detections(log_path, args.minutes)
    detection_stats = analyze_detections(recent_detections)

    # Check snapshots
    snapshot_stats = check_snapshots(snapshot_dir) if snapshot_dir else None

    # Compile report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_period_minutes": args.minutes,
        "detections": detection_stats,
        "snapshots": snapshot_stats
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        # Pretty print
        print("\n" + "=" * 60)
        print("FACE DETECTION SYSTEM STATUS")
        print("=" * 60)
        print(f"Report Time: {report['timestamp']}")
        print(f"Analysis Period: Last {args.minutes} minutes")
        print()

        print("DETECTION STATISTICS:")
        print("-" * 40)
        ds = detection_stats
        print(f"  Status: {ds['status']}")
        if ds['status'] != 'NO_DATA':
            print(f"  Sensor ID: {ds['sensor_id']}")
            print(f"  Log Entries: {ds['log_entries']}")
            print(f"  Total Detections: {ds['total_detections']}")
            print(f"  Unique Persons: {ds['unique_persons']}")
            print(f"  Average Confidence: {ds['average_confidence']:.1%}")
            print(f"  Average Age: {ds['average_age']}")
            print(f"  Gender Distribution: {ds['gender_distribution']}")
            print(f"  Time Range: {ds['time_range']}")

        if snapshot_stats:
            print()
            print("SNAPSHOTS:")
            print("-" * 40)
            print(f"  Status: {snapshot_stats['status']}")
            print(f"  Recent Count: {snapshot_stats['count']}")
            if snapshot_stats['latest']:
                print("  Latest snapshots:")
                for snap in snapshot_stats['latest'][:3]:
                    print(f"    - {snap['name']} ({snap['time']}, {snap['size_kb']}KB)")

        print()
        print("=" * 60)

        # Health check summary
        if ds['status'] == 'NO_DATA':
            print("WARNING: No detections in the last hour!")
            print("Check if the system is running.")
        elif ds['status'] == 'LOW_CONFIDENCE':
            print("WARNING: Average confidence is low (<50%)")
            print("Consider adjusting camera position or lighting.")
        else:
            print("System appears to be working normally.")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

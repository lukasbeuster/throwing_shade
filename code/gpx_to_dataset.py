"""
gpx_to_dataset.py

Convert a folder of GPX track files into two CSV datasets:
- trackpoints.csv: columns (tripUuid, time, latitude, longitude, elevation)
- metadata.csv: columns (tripUuid, totalSteps, calories, isManual, activityType, userId, elapsedTime, utcOffset, hasPoints, startTime, endTime, entryType)
"""

import os
import glob
import gpxpy
import pandas as pd
import csv
from pathlib import Path
from tqdm import tqdm


def parse_desc(desc_text):
    """
    Parse the <desc> block into a metadata dict.
    """
    metadata = {}
    for line in desc_text.strip().splitlines():
        if '=' in line:
            key, val = line.split('=', 1)
            metadata[key.strip()] = val.strip()
    return metadata


def process_gpx_folder(folder_path, output_dir, points_filename='trackpoints.csv', metadata_filename='metadata.csv'):
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    points_path = output_path / points_filename
    metadata_path = output_path / metadata_filename

    gpx_files = glob.glob(os.path.join(folder_path, '*.gpx'))
    total_files = len(gpx_files)
    # Prepare CSV writers for streaming output
    with open(metadata_path, 'w', newline='') as meta_f, open(points_path, 'w', newline='') as pts_f:
        meta_writer = csv.DictWriter(meta_f, fieldnames=[
            'tripUuid', 'totalSteps', 'calories', 'isManual', 'activityType', 'userId',
            'elapsedTime', 'utcOffset', 'hasPoints', 'startTime', 'endTime', 'entryType'
        ])
        pts_writer = csv.DictWriter(pts_f, fieldnames=[
            'tripUuid', 'time', 'latitude', 'longitude', 'elevation'
        ])
        meta_writer.writeheader()
        pts_writer.writeheader()
        trip_count = 0
        point_count = 0
        print(f"Starting processing of {total_files} GPX files...")
        for filepath in tqdm(gpx_files, desc="Processing GPX files", total=total_files, unit="file"):
            with open(filepath, 'r') as f:
                gpx = gpxpy.parse(f)

            for track in gpx.tracks:
                # Extract and parse metadata from the <desc> tag
                desc = track.description or ''
                meta = parse_desc(desc)
                trip_uuid = meta.get('tripUuid', os.path.splitext(os.path.basename(filepath))[0])

                # Write metadata (one entry per trip)
                meta_writer.writerow({
                    'tripUuid': trip_uuid,
                    'totalSteps': meta.get('totalSteps'),
                    'calories': meta.get('calories'),
                    'isManual': meta.get('isManual'),
                    'activityType': meta.get('activityType'),
                    'userId': meta.get('userId'),
                    'elapsedTime': meta.get('elapsedTime'),
                    'utcOffset': meta.get('utcOffset'),
                    'hasPoints': meta.get('hasPoints'),
                    'startTime': meta.get('startTime'),
                    'endTime': meta.get('endTime'),
                    'entryType': meta.get('entryType'),
                })
                trip_count += 1
                tqdm.write(f"Processed {trip_count} trips so far ({filepath})")

                # Write each trackpoint
                for segment in track.segments:
                    for pt in segment.points:
                        pts_writer.writerow({
                            'tripUuid': trip_uuid,
                            'time': pt.time.isoformat() if pt.time else '',
                            'latitude': pt.latitude,
                            'longitude': pt.longitude,
                            'elevation': pt.elevation,
                        })
                        point_count += 1

    print(f"Processed {trip_count} trips.")
    print(f"Generated {point_count} trackpoints.")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved trackpoints to {points_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert GPX folder to datasets')
    parser.add_argument('folder', help='Path to GPX files directory')
    parser.add_argument('--points', default='trackpoints.csv', help='Output trackpoints CSV filename')
    parser.add_argument('--metadata', default='metadata.csv', help='Output metadata CSV filename')
    parser.add_argument('--output-dir', default='.', help='Directory to save output CSVs')
    args = parser.parse_args()

    process_gpx_folder(args.folder, args.output_dir, args.points, args.metadata)

#!/usr/bin/env python3
"""
Example of running by invoking with one photo at a time
"""

import cProfile
import argparse
import os
import posixpath
import json
from scipy import ndimage

import process, solve
from common import util
from common.config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='Path to a directory of JPEGs')
    parser.add_argument('--working-dir', help='Where to process into')
    args = parser.parse_args()
    _prepare_new_run(args.working_dir)

    # Open the batch.json file containing the robot position each photo was taken at
    batch_info_file = posixpath.join(args.input_path, "batch.json")
    with open(batch_info_file, "r") as jsonfile:
        batch_info = json.load(jsonfile)

    piece_id = 1
    for f in os.listdir(args.input_path):
        if f.endswith('.jpg') or f.endswith('.jpeg'):
            print(f"{util.YELLOW}### Processing {f} ###{util.WHITE}")
            # Obtain robot position where this photo was taken, from the json file
            x,y,z = [d['position'] for d in batch_info['photos'] if d['file_name'] == f][0]
            robot_state = dict(photo_at_motor_position=[x,y,z])
            # Process photo
            piece_id = process.process_photo(
                photo_path=os.path.join(args.input_path, f), 
                working_dir=args.working_dir, 
                starting_piece_id=piece_id, 
                robot_state=robot_state
            )

    print("Solving")
    solve.solve(path=args.working_dir)


def _prepare_new_run(path):
    for d in [PHOTO_BMP_DIR, SEGMENT_DIR, VECTOR_DIR, DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR]:
        os.makedirs(os.path.join(path, d), exist_ok=True)
        for f in os.listdir(os.path.join(path, d)):
            os.remove(os.path.join(path, d, f))


if __name__ == '__main__':
    PROFILE = False
    if PROFILE:
        cProfile.run('main()', 'profile_results.prof')
    else:
        main()

"""
Example of running by invoking with one photo at a time
"""

import cProfile
import argparse
import os

import process, solve
from common import util
from common.config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='Path to a directory of JPEGs')
    parser.add_argument('--working-dir', help='Where to process into')
    args = parser.parse_args()
    _prepare_new_run(args.working_dir)

    piece_id = 1
    for f in os.listdir(args.input_path):
        if f.endswith('.jpg') or f.endswith('.jpeg'):
            print(f"{util.YELLOW}### Processing {f} ###{util.WHITE}")
            robot_state = {}  # TODO
            piece_id = process.process_photo(photo_path=os.path.join(args.input_path, f), working_dir=args.working_dir, starting_piece_id=piece_id, robot_state=robot_state)

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

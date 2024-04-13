"""
Processes photographs of pieces into digitzed piece data
"""

import os
import time
import multiprocessing
import re

from common import bmp, extract, util, vector
from common.config import *


def process_photo(photo_path, working_dir, starting_piece_id, robot_state):
    """
    Takes in a path to a photo of a part of the bed and the robot's state when the photo was taken
    Processes that photo into digital puzzle piece information
    Stores intermediate results in the working directory
    TODO: make use of robot_state

    Returns the final piece ID
    """
    # 1 - segment into a binary BMP
    bmp_path = os.path.join(working_dir, PHOTO_BMP_DIR, f'{os.path.basename(photo_path).split(".")[0]}.bmp')
    bmp.photo_to_bmp(args=(photo_path, bmp_path))

    # 2 - extract pieces from the binary BMP
    extract_path = os.path.join(working_dir, SEGMENT_DIR)
    extracted_paths = extract.extract_pieces(args=(bmp_path, extract_path))

    # 3 - vectorize the pieces from each of the extracted bitmaps
    piece_id = starting_piece_id
    for f in extracted_paths:
        vector_path = os.path.join(working_dir, VECTOR_DIR)
        vector.load_and_vectorize(args=(f, piece_id, vector_path, False))
        piece_id += 1

    return piece_id


def batch_process_photos(path, serialize, id=None, start_at_step=0, stop_before_step=3):
    """
    Given a path to a working directory that contains a 0_input subdirectory full of photos
    Batch processes them into digital puzzle piece information

    start_at_step: the step to start processing at
    stop_before_step: the step to stop processing at
    id: only process the photo with this ID
    """
    if start_at_step <= 0 and stop_before_step > 0:
        _bmp_all(input_path=os.path.join(path, PHOTOS_DIR), output_path=os.path.join(path, PHOTO_BMP_DIR), id=id)

    if start_at_step <= 1 and stop_before_step > 1:
        _extract_all(input_path=os.path.join(path, PHOTO_BMP_DIR), output_path=os.path.join(path, SEGMENT_DIR), id=id)

    if start_at_step <= 2 and stop_before_step > 2:
        _vectorize(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, VECTOR_DIR), id=id, serialize=serialize)


def _bmp_all(input_path, output_path, id):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    print(f"\n{util.RED}### 0 - Segmenting photos into binary images ###{util.WHITE}\n")

    if id:
        fs = [f'{id}.jpeg']
    else:
        fs = [f for f in os.listdir(input_path) if re.match(r'.*\.jpe?g', f)]

    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        output_name = f.split('.')[0]
        output_img_path = os.path.join(output_path, f'{output_name}.bmp')
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(bmp.photo_to_bmp, args)


def _extract_all(input_path, output_path, id):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    print(f"\n{util.RED}### 1 - Extracting pieces from photo bitmaps ###{util.WHITE}\n")

    if id:
        fs = [f'{id}.bmp']
    else:
        fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]

    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        args.append([input_img_path, output_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(extract.extract_pieces, args)


def _vectorize(input_path, output_path, id, serialize):
    """
    Loads each image.bmp in the input directory, converts it to an SVG in the output directory
    """
    print(f"\n{util.RED}### 2 - Vectorizing ###{util.WHITE}\n")

    start_time = time.time()
    i = id if id is not None else 1

    args = []

    for f in os.listdir(input_path):
        if not f.endswith('.bmp'):
            continue
        path = os.path.join(input_path, f)
        render = (i == id)
        args.append([path, i, output_path, render])

        i += 1
        if id is not None:
            break

    if serialize:
        for arg in args:
            vector.load_and_vectorize(arg)
    else:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.map(vector.load_and_vectorize, args)

    duration = time.time() - start_time
    print(f"Vectorizing took {round(duration, 2)} seconds ({round(duration /i, 2)} seconds per piece)")

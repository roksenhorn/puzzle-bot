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

    Returns the final piece ID
    """
    metadata = {
        "robot_state": robot_state,
    }

    # 1 - segment into a binary BMP
    bmp_path = os.path.join(working_dir, PHOTO_BMP_DIR, f'{os.path.basename(photo_path).split(".")[0]}.bmp')
    width, height, scale_factor = bmp.photo_to_bmp(args=(photo_path, bmp_path))
    metadata['scale_factor'] = scale_factor
    metadata['bmp_width'] = width
    metadata['bmp_height'] = height
    metadata["original_photo_name"] = os.path.basename(photo_path)

    # 2 - extract pieces from the binary BMP
    extract_path = os.path.join(working_dir, SEGMENT_DIR)
    extracted_paths, extracted_photo_space_positions = extract.extract_pieces(args=(bmp_path, extract_path, scale_factor))

    # 3 - vectorize the pieces from each of the extracted bitmaps
    piece_id = starting_piece_id
    args = []
    for f in extracted_paths:
        piece_metadata = metadata.copy()
        photo_space_position = extracted_photo_space_positions[f]
        piece_metadata["photo_space_origin"] = photo_space_position
        vector_path = os.path.join(working_dir, VECTOR_DIR)
        args.append((f, piece_id, vector_path, piece_metadata, photo_space_position, scale_factor, False))
        piece_id += 1

    SERIALIZE = False  # flip this for improved debugability
    if SERIALIZE:
        print("!!!!! RUNNING IN SERIAL MODE - Only do this if you're debugging. Flip back to parallel mode for 10x speedup !!!!!")
        for arg in args:
            vector.load_and_vectorize(arg)
    else:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.map(vector.load_and_vectorize, args)

    return piece_id


def batch_process_photos(path, serialize, id=None, start_at_step=0, stop_before_step=3):
    """
    Given a path to a working directory that contains a 0_input subdirectory full of photos
    Batch processes them into digital puzzle piece information

    start_at_step: the step to start processing at
    stop_before_step: the step to stop processing at
    id: only process the photo with this ID
    """
    metadata = { "robot_state": {} }

    if start_at_step <= 0 and stop_before_step > 0:
        width, height, scale_factor = _bmp_all(input_path=os.path.join(path, PHOTOS_DIR), output_path=os.path.join(path, PHOTO_BMP_DIR), id=id)
    else:
        # mock when skipping step 0
        width, height, scale_factor = 1, 1, 1.0

    metadata['scale_factor'] = scale_factor
    metadata['bmp_width'] = width
    metadata['bmp_height'] = height

    photo_space_positions = {}
    if start_at_step <= 1 and stop_before_step > 1:
        output = _extract_all(input_path=os.path.join(path, PHOTO_BMP_DIR), output_path=os.path.join(path, SEGMENT_DIR), scale_factor=scale_factor, id=id)
        photo_space_positions_list = [o[1] for o in output]
        for d in photo_space_positions_list:
            photo_space_positions.update(d)
    else:
        # mock when skipping step 1
        for f in os.listdir(os.path.join(path, SEGMENT_DIR)):
            photo_space_positions[os.path.join(path, SEGMENT_DIR, f)] = (0, 0)

    if start_at_step <= 2 and stop_before_step > 2:
        _vectorize(input_path=os.path.join(path, SEGMENT_DIR), metadata=metadata, output_path=os.path.join(path, VECTOR_DIR), photo_space_positions=photo_space_positions, scale_factor=scale_factor, id=id, serialize=serialize)


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
        # capture the output from each call to photo_to_bmp
        output = pool.map(bmp.photo_to_bmp, args)

    return output[0]


def _extract_all(input_path, output_path, scale_factor, id):
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
        args.append([input_img_path, output_path, scale_factor])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        output = pool.map(extract.extract_pieces, args)

    return output


def _vectorize(input_path, output_path, metadata, photo_space_positions, scale_factor, id, serialize):
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
        photo_space_position = photo_space_positions[path]
        piece_metadata = metadata.copy()
        piece_metadata["photo_space_origin"] = photo_space_position
        piece_metadata["original_photo_name"] = f
        args.append([path, i, output_path, piece_metadata, photo_space_position, scale_factor, render])

        if id is not None:
            break
        else:
            i += 1

    if serialize:
        for arg in args:
            vector.load_and_vectorize(arg)
    else:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.map(vector.load_and_vectorize, args)

    duration = time.time() - start_time
    print(f"Vectorizing took {round(duration, 2)} seconds ({round(duration /i, 2)} seconds per piece)")

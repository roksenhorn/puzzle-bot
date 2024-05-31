"""
Processes photographs of pieces into digitzed piece data
"""

import os
import time
import multiprocessing
import re
import pathlib

from common import bmp, extract, util, vector, dedupe
from common.config import *


def batch_process_photos(path, serialize, robot_states, id=None, start_at_step=0, stop_before_step=3):
    """
    Given a path to a working directory that contains a 0_input subdirectory full of photos
    Batch processes them into digital puzzle piece information

    robot_states: a dictionary of photo filenames to robot states
    start_at_step: the step to start processing at
    stop_before_step: the step to stop processing at
    id: only process the photo with this ID
    """

    if start_at_step <= 0 and stop_before_step > 0:
        width, height, scale_factor = _bmp_all(input_path=pathlib.Path(path).joinpath(PHOTOS_DIR), output_path=pathlib.Path(path).joinpath(PHOTO_BMP_DIR), id=id)
    else:
        # we'll need realistic data when skipping, so do the minimum amount of work
        input_dir = pathlib.Path(path).joinpath(PHOTOS_DIR)
        f = [f for f in os.listdir(input_dir) if re.match(r'.*\.jpe?g', f)][0]
        args = [pathlib.Path(input_dir).joinpath(f), "C:/Temp/trash.bmp"]
        width, height, scale_factor = bmp.photo_to_bmp(args)
        print(f"BMPs are {width}x{height} @ scale {scale_factor}")

    metadata = {
        "robot_state": {},  # will get filled in for each piece when vectorizing
        "scale_factor": scale_factor,
        "bmp_width": width,
        "bmp_height": height,
        "photo_width": width * scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[1] + CROP_TOP_RIGHT_BOTTOM_LEFT[3],
        "photo_height": height * scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[0] + CROP_TOP_RIGHT_BOTTOM_LEFT[2],
    }

    photo_space_positions = {}
    if start_at_step <= 1 and stop_before_step > 1:
        photo_space_positions = _extract_all(input_path=pathlib.Path(path).joinpath(PHOTO_BMP_DIR), output_path=pathlib.Path(path).joinpath(SEGMENT_DIR), scale_factor=scale_factor)
    else:
        # mock when skipping step 1
        for f in os.listdir(pathlib.Path(path).joinpath(SEGMENT_DIR)):
            photo_space_positions[f] = (0, 0)

    if start_at_step <= 2 and stop_before_step > 2:
        dedupe.bmp_deduplicate(path=path, output_path=pathlib.Path(path).joinpath(DEDUPED_DIR))

    if start_at_step <= 3 and stop_before_step > 3:
        _vectorize_all(input_path=pathlib.Path(path).joinpath(SEGMENT_DIR), metadata=metadata, robot_states=robot_states, output_path=pathlib.Path(path).joinpath(VECTOR_DIR), photo_space_positions=photo_space_positions, scale_factor=scale_factor, id=id, serialize=serialize)


def _bmp_all(input_path, output_path, id):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    print(f"\n{util.BLUE}### 0 - Segmenting photos into binary images ###{util.WHITE}\n")

    if id:
        fs = [f'{id}.jpeg']
    else:
        fs = [f for f in os.listdir(input_path) if re.match(r'.*\.jpe?g', f)]

    args = []
    for f in fs:
        input_img_path = pathlib.Path(input_path).joinpath(f)
        output_name = f.split('.')[0]
        output_img_path = pathlib.Path(output_path).joinpath(f'{output_name}.bmp')
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # capture the output from each call to photo_to_bmp
        output = pool.map(bmp.photo_to_bmp, args)

    return output[0]


def _extract_all(input_path, output_path, scale_factor):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    print(f"\n{util.BLUE}### 1 - Extracting pieces from photo bitmaps ###{util.WHITE}\n")
    start_time = time.time()
    output = extract.batch_extract(input_path, output_path, scale_factor)
    duration = time.time() - start_time
    print(f"Extracted {len(output)} pieces in {round(duration, 2)} seconds")
    return output


def _vectorize_all(input_path, output_path, metadata, robot_states, photo_space_positions, scale_factor, id, serialize):
    """
    Loads each image.bmp in the input directory, converts it to an SVG in the output directory
    """
    print(f"\n{util.BLUE}### 3 - Vectorizing ###{util.WHITE}\n")

    start_time = time.time()
    i = id if id is not None else 1

    args = []

    for f in os.listdir(input_path):
        if not f.endswith('.bmp'):
            continue

        path = pathlib.Path(input_path).joinpath(f)
        render = (i == id)
        photo_space_position = photo_space_positions[f]
        original_photo_name = '_'.join(f.split('.')[0].split('_')[:-1]) + ".jpg"  # reverse engineer the BMP name to the JPG
        piece_metadata = metadata.copy()
        piece_metadata["photo_space_origin"] = photo_space_position
        piece_metadata["original_photo_name"] = original_photo_name
        piece_metadata["robot_state"] = {"photo_at_motor_position": robot_states[original_photo_name]}
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

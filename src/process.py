"""
Processes photographs of pieces into digitzed piece data
"""

import cProfile
import argparse
import os
import time
import multiprocessing
import re

from common import bmp, board, connect, dedupe, extract, util, vector
from common.config import *


def batch_process_photos(path, serialize, id, start_at_step, stop_before_step):
    start_time = time.time()

    if start_at_step <= 0 and stop_before_step > 0:
        bmp_all(input_path=os.path.join(path, PHOTOS_DIR), output_path=os.path.join(path, PHOTO_BMP_DIR), id=id)

    if start_at_step <= 1 and stop_before_step > 1:
        extract_all(input_path=os.path.join(path, PHOTO_BMP_DIR), output_path=os.path.join(path, SEGMENT_DIR), id=id)

    if start_at_step <= 2 and stop_before_step > 2:
        vectorize(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, VECTOR_DIR), id=id, serialize=serialize)

    duration = time.time() - start_time
    print(f"\n\n{util.GREEN}### Batch processed photos in {round(duration, 2)} sec ###{util.WHITE}\n")


def bmp_all(input_path, output_path, id):
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


def extract_all(input_path, output_path, id):
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
        output_img_path = os.path.join(output_path)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(extract.extract_pieces, args)


def vectorize(input_path, output_path, id, serialize):
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

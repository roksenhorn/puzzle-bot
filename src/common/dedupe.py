import os
import json
import numpy as np
import math
from glob import glob
import shutil
from pathlib import Path
import multiprocessing
import re
import PIL

from common import util, sides
from common.config import *


BMP_DUPLICATE_THRESHOLD = 3.0
THUMBNAIL_SIZE = 120

DUPLICATE_THRESHOLD = 2.2
SIDE_MISALIGNMENT_RAD = 15.0 * math.pi / 180


def bmp_deduplicate(path, output_path):
    os.makedirs(os.path.join(path, "3a_thumbnails"), exist_ok=True)
    os.makedirs(os.path.join(path, "3b_filled"), exist_ok=True)

    thumbnail(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, "3a_thumbnails"))
    fill_islands(input_path=os.path.join(path, "3a_thumbnails"), output_path=os.path.join(path, "3b_filled"))
    ssd(input_path=os.path.join(path, "3b_filled"), original_path=os.path.join(path, SEGMENT_DIR), output_path=output_path)


def fill_islands(input_path, output_path):
    print("Filling islands...")
    fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(_fill_islands, args)


def _fill_islands(args):
    input_path, output_path = args
    pixels, width, height = util.load_bmp_as_binary_pixels(input_path)
    util.remove_tiny_islands(pixels, ignore_islands_along_border=True, island_value=0)

    img = PIL.Image.new('1', (width, height))
    img.putdata([pixel for row in pixels for pixel in row])
    img.save(output_path)


def thumbnail(input_path, output_path):
    print("Creating thumbnails...")
    fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    args = []
    for f in fs:
        # load the image and save off a thumbnail version scaled down to 20% then padded to 60x60
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(_thumbnail, args)


def _thumbnail(args):
    input_path, output_path = args
    img = PIL.Image.open(input_path)
    w, h = img.size[0] // 10, img.size[1] // 10
    img.thumbnail((w, h), PIL.Image.NEAREST)
    img = img.convert('1')

    # Calculate the padding size
    padding_width = (THUMBNAIL_SIZE - img.size[0]) // 2
    padding_height = (THUMBNAIL_SIZE - img.size[1]) // 2

    # Create a new image with the desired size and paste the thumbnail in the center
    padded_img = PIL.Image.new('1', (THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    padded_img.paste(img, (padding_width, padding_height))
    padded_img.save(output_path)


def ssd(input_path, original_path, output_path):
    print("Running SSD between all thumbnails...")
    fs = [os.path.join(input_path, f) for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    images = [util.load_bmp_as_binary_pixels(f)[0] for f in fs]

    dupes = set()
    keeps = set()
    debug = []

    for i, img in enumerate(images):
        if i in dupes:
            continue
        else:
            keeps.add(i)

        for j, other_img in enumerate(images):
            if i == j:
                continue
            ssd_score = util.normalized_ssd(img, other_img)
            if ssd_score < BMP_DUPLICATE_THRESHOLD:
                fi = fs[i].split('/')[-1].split('.')[0]
                fj = fs[j].split('/')[-1].split('.')[0]
                dupes.add(j)
                debug.append((fi, fj, ssd_score))

    print(f"Starting with {len(images)} images")
    print(f"Removing {len(dupes)} images")
    print(f"Resulting in {len(keeps)} images")

    for i in keeps:
        f = fs[i].split(os.path.sep)[-1]
        input_img_path = Path(original_path).joinpath(f)
        output_img_path = Path(output_path).joinpath(f)
        shutil.copy(input_img_path, output_img_path)

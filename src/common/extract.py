import os
import PIL
import numpy as np
import subprocess

from common import util
from common.config import *


def batch_extract(photo_paths, output_path, scale_factor):
    # gcc -o find_islands find_islands.c
    cmd = f"gcc -o find_islands.o {os.path.join(os.path.dirname(__file__), '../c/find_islands.c')}"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)

    exit(1)


def extract_pieces(args):
    """
    Returns how many pieces were extracted
    """
    input_path, output_path, scale_factor = args

    print(f"> Extracting pieces from {input_path.split('/')[-1]}")
    pixels, _, _ = util.load_bmp_as_binary_pixels(input_path)

    islands = util.find_islands(pixels, min_island_area=MIN_PIECE_AREA, ignore_islands_along_border=True)
    output_paths = []
    output_photo_space_positions = {}

    piece_id = 1
    for island, origin in islands:
        unique_id = input_path.split(os.path.sep)[-1].split('.')[0]
        piece_output_path = os.path.join(output_path, f'{unique_id}_{piece_id}.bmp')
        ok, photo_space_position = _clean_and_save_piece(island, piece_output_path, origin, scale_factor)
        if ok:
            output_paths.append(piece_output_path)
            output_photo_space_positions[piece_output_path] = photo_space_position
            piece_id += 1

    print(f"> Extracted {len(output_paths)} pieces from {input_path.split('/')[-1]}")
    return output_paths, output_photo_space_positions


def _clean_and_save_piece(pixels, output_path, origin, scale_factor):
    """
    Pad the piece with a border, clean it a bit, and save it as a BMP
    Returns False if the piece was rejected
    """
    # reject islands that are really narrow, like a seem along an edge that was picked up incorrectly
    w, h = pixels.shape
    if w < 0.25 * h or h < 0.25 * w:
        # print(f"Skipping piece {piece_id} because it is too thin")
        return False, None

    # pad the pixels with a black border for processing
    pixels = np.pad(pixels, pad_width=1, mode='constant', constant_values=0)

    # clean up any thin strands of pixels like hairs or dust
    util.remove_stragglers(pixels)

    # remove any tiny islands that are probably noise
    util.remove_tiny_islands(pixels)

    # trim the piece to the smallest bounding box
    rows = np.any(pixels, axis=1)
    cols = np.any(pixels, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pixels = pixels[rmin:rmax+1, cmin:cmax+1]

    # re-pad the pixels with a black border
    PAD_BY = 1
    pixels = np.pad(pixels, pad_width=PAD_BY, mode='constant', constant_values=0)

    # unscale the origin and uncrop the photo
    photo_space_position = (origin[0] / scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[-1], origin[1] / scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[0])

    # save a binary bitmpa
    w, h = pixels.shape
    img = PIL.Image.new('1', (h, w))
    img.putdata(pixels.flatten())
    img.save(output_path)
    return True, photo_space_position
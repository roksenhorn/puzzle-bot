import time
from PIL import Image
from typing import List, Tuple

from common import util
from common.config import *


def photo_to_bmp(args):
    input_photo_filename, output_bmp_filename = args
    return segment(input_photo_filename, output_bmp_filename)


def segment(input_photo_filename, output_path=None, width=BMP_WIDTH, threshold=SEG_THRESH, crop=True):
    """
    Takes in a photo of one or more puzzle pieces
    Generates a binary image that is slightly cleaned up
    Removes dust and debris from the image
    Scales the binary image by the provided factor

    If an output path is provided, this function saves the resulting bitmap
    Returns the pixels and dimensions

    width: the maximum width of the output image
    white_pieces: whether the pieces are white on a black background (True) or black on a white background (False)
    threshold: the threshold for the binary image
    clean: whether to clean up the image iwth some post-processing
    """
    print(f"> Segmenting photo `{input_photo_filename}` into `{output_path}`")
    bw_pixels, width, height, scale_factor = util.binary_pixel_data_for_photo(input_photo_filename,
                                                                              threshold=threshold, max_width=width,
                                                                              # cropping will mess with the scaling math that we use later on
                                                                              crop=0)
    if output_path:
        _save(output_path, bw_pixels, width, height)

    return scale_factor


def _save(output_path, bw_pixels, width, height):
    img = Image.new('1', (width, height))
    img.putdata([pixel for row in bw_pixels for pixel in row])
    img.save(output_path)

"""
Given a path to an image, counts the number of white and black pixels in the image.
"""

import argparse
from common import util
from common.config import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', help='Path to the image')
    args = parser.parse_args()

    binary_pixels, _, _, _ = util.binary_pixel_data_for_photo(args.path, threshold=150)
    white = sum(sum(binary_pixels == 1))
    black = sum(sum(binary_pixels == 0))
    percent_white = white / (white + black)
    print(f"White: {white} \t Black: {black} \t % white: {percent_white}")

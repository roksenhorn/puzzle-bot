from PIL import Image
from typing import List, Tuple

from common import util


SEG_THRESH = (125*3)


def segment(input_photo_filename, output_path=None, width=1500, white_pieces=True, threshold=SEG_THRESH, clean=True):
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
    bw_pixels, width, height = util.binary_pixel_data_for_photo(input_photo_filename, white_pieces=white_pieces, max_width=width)
    if clean:
        _clean(bw_pixels, width, height)
    if output_path:
        _save(output_path, bw_pixels, width, height)
    return (bw_pixels, width, height)


def _clean(bw_pixels, width, height):
    _remove_small_islands(bw_pixels)
    _remove_stragglers(bw_pixels, width, height)


def _remove_small_islands(pixels) -> None:
    """
    Find and remove all islands of pixels that are smaller than some set number of pixels
    """
    # find all the islands
    lines = [[e for e in l] for l in pixels]
    islands = util.find_islands(lines, ignore_islands_along_border=False)

    # sort islands (which is a list of list) by len of each island
    islands.sort(key=lambda i: len(i), reverse=True)

    # remove all islands that are less than 1/4 the size of the biggest island
    min_size = len(islands[0]) // 4
    print(f"Removing islands smaller than {min_size} pixels")

    # remove all other islands
    removed_count = 0
    for island in islands:
        if len(island) < min_size:
            removed_count += 1
            for x, y in island:
                pixels[y][x] = 0

    print(f"Removed {removed_count} tiny islands")


def _remove_stragglers(pixels, width, height) -> bool:
    """
    Removes any pixels that are only connected to one other piece
    Returns True if any were removed
    """
    removed = False

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            v = pixels[y][x]
            if v != 1:
                continue

            # All 8 neighbors
            above_left = pixels[y - 1][x - 1]
            above = pixels[y - 1][x]
            above_right = pixels[y - 1][x + 1]
            right = pixels[y][x + 1]
            below_right = pixels[y + 1][x + 1]
            below = pixels[y + 1][x]
            below_left = pixels[y + 1][x - 1]
            left = pixels[y][x - 1]
            neighbors = [
                above_left,
                above,
                above_right,
                right,
                below_right,
                below,
                below_left,
                left,
            ]
            borders = [True for n in neighbors if n == 1]
            if len(borders) <= 1:
                # straggler only connected by one
                pixels[y][x] = 0
                removed = True

            # if there are only 2 neighbors, and they are not adjacent (e.g. no [1, 1] subset in the list), then these
            # are one-pixel-wide bridges that should be removed
            if len(borders) == 2 and not util.sublist_exists(borders, [1, 1]):
                pixels[y][x] = 0
                removed = True

    if removed:
        return _remove_stragglers(pixels, width, height)

    return False


def _save(output_path, bw_pixels, width, height):
    img = Image.new('1', (width, height))
    img.putdata([pixel for row in bw_pixels for pixel in row])
    img.save(output_path)

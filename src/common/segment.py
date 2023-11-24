from PIL import Image
from typing import List, Tuple

from common import util


SEG_THRESH = (125*3)
SCALE_BY = 0.12


def flip(input_path, output_path):
    """
    Loads an image file, flips it horizontally, and saves it to the output path
    """
    with Image.open(input_path) as img:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(output_path)


def segment(filename, output_path=None, scale_by=SCALE_BY,
            threshold=SEG_THRESH, clean_and_crop=True):
    """
    Extracts a darker puzzle piece from a bright background
    Removes dust and debris from the image
    Scales the binary image by the provided factor

    Pixel values of 1 = puzzle piece
    Pixel values of 0 = background

    If an output path is provided, this function saves the resulting bitmap
    Returns the pixels and dimensions
    """
    rgb_pixels, width, height = _load(filename, scale_by)
    bw_pixels = _segment(rgb_pixels, width, height, threshold)
    if clean_and_crop:
        width, height = _clean_and_crop(bw_pixels, width, height)
    if output_path:
        _save(output_path, bw_pixels, width, height)
    return (bw_pixels, width, height)


def _load(filename, scale_by) -> List[Tuple[int, int, int]]:
    """
    Loads an image file and returns a list of pixels, where each pixel is a tuple of (r, g, b) values
    """
    with Image.open(filename) as img:
        width, height = img.size

        # resize to a maximum dimension
        width = round(scale_by * width)
        height = round(scale_by * height)
        img = img.resize((width, height))

        rgb_pixels = list(img.getdata())

    return rgb_pixels, width, height


def _segment(rgb_pixels, width, height, threshold) -> None:
    """
    Segments the image into black and white pixels
    White is a part of the puzzle piece
    Black is background
    """
    bw_pixels = []

    # Convert pixels to 0 or 1
    for i, (pixel_r, pixel_g, pixel_b) in enumerate(rgb_pixels):
        x = i % width
        y = i // width
        if y >= len(bw_pixels):
            bw_pixels.append([])
        row = bw_pixels[y]
        brightness = 0 if (pixel_r + pixel_g + pixel_b) > threshold else 1
        row.append(brightness)
        bw_pixels[y] = row

    return bw_pixels


def _clean_and_crop(bw_pixels, width, height):
    _remove_small_islands(bw_pixels)
    _remove_stragglers(bw_pixels, width, height)
    width, height = _crop(bw_pixels, width, height)
    return width, height


def _remove_small_islands(pixels) -> None:
    """
    Find and remove all islands of pixels that are smaller than some set number of pixels
    """
    # find all the islands
    lines = [[e for e in l] for l in pixels]
    islands = util.find_islands(lines)

    # sort islands (which is a list of list) by len of each island
    islands.sort(key=lambda i: len(i), reverse=True)

    # keep the biggest
    islands.pop(0)

    # remove all other islands
    for island in islands:
        for x, y in island:
            pixels[y][x] = 0


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


def _crop(pixels, width, height) -> None:
    # crop from the top
    while not any(pixels[1]):
        pixels.pop(1)
        height -= 1

    # crop from the bottom
    while not any(pixels[-2]):
        pixels.pop(-2)
        height -= 1

    # crop from the left
    continue_cropping_left = True
    while continue_cropping_left:
        for y in range(height):
            if pixels[y][1] == 1:
                continue_cropping_left = False
                break
        if continue_cropping_left:
            for y in range(height):
                pixels[y] = pixels[y][1:]
            width -= 1

    # crop from the right
    continue_cropping_right = True
    while continue_cropping_right:
        for y in range(height):
            if pixels[y][-2] == 1:
                continue_cropping_right = False
                break
        if continue_cropping_right:
            for y in range(height):
                pixels[y] = pixels[y][:-1]
            width -= 1

    return width, height


def _save(output_path, bw_pixels, width, height):
    img = Image.new('1', (width, height))
    img.putdata([pixel for row in bw_pixels for pixel in row])
    img.save(output_path)

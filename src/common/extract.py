import os
import PIL

from common import util


PIL.Image.MAX_IMAGE_PIXELS = 912340000

MIN_PIECE_AREA = 100 * 100


def extract_pieces(args):
    """
    Returns how many pieces were extracted
    """
    input_path, output_path, unique_id = args
    pixels, _, _ = util.load_bmp_as_binary_pixels(input_path)

    def found_island(island, i):
        return _clean_and_save_piece(unique_id, i + 1, island, output_path)

    islands = util.find_islands(pixels, callback=found_island, ignore_islands_along_border=True)
    print(f"Extracted {len(islands)} pieces from {input_path.split('/')[-1]}")
    return len(islands)


def _clean_and_save_piece(unique_id, piece_id, piece_coordinates, output_path):
    # reject noise in the image that is clearly too small to be a piece
    if len(piece_coordinates) < MIN_PIECE_AREA:
        return False

    xs = [x for (x, _) in piece_coordinates]
    ys = [y for (_, y) in piece_coordinates]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    width = maxx - minx + 1
    height = maxy - miny + 1

    # pad the image with a border
    BORDER_WIDTH_PX = 1
    width += (2 * BORDER_WIDTH_PX)
    height += (2 * BORDER_WIDTH_PX)

    # reject islands that are really narrow, like a seem along an edge that was picked up incorrectly
    if width < 0.26 * height or height < 0.25 * width:
        # print(f"Skipping piece {piece_id} because it is too thin")
        return False

    pixels = []
    for i in range(height):
        pixels.append([])
        for j in range(width):
            pixels[i].append(0) # start with an all black background

    for (x, y) in piece_coordinates:
        xx = x - minx + BORDER_WIDTH_PX
        yy = y - miny + BORDER_WIDTH_PX
        pixels[yy][xx] = 1

    # clean up any thin strands of pixels like hairs or dust
    util.remove_stragglers(pixels, width, height)

    img = PIL.Image.new('1', (width, height))
    img.putdata([pixel for row in pixels for pixel in row])
    img.save(os.path.join(output_path, f'{unique_id}_{piece_id}.bmp'))
    return True

import os
import PIL

from common import util


MIN_PIECE_AREA = 100 * 100


def extract_pieces(args):
    """
    Returns how many pieces were extracted
    """
    input_path, output_path = args
    pixels, _, _ = util.load_bmp_as_binary_pixels(input_path)

    islands = util.find_islands(pixels, ignore_islands_along_border=True)
    output_paths = []

    piece_id = 1
    for island in islands:
        unique_id = input_path.split('/')[-1].split('.')[0]
        piece_output_path = os.path.join(output_path, f'{unique_id}_{piece_id}.bmp')
        if _clean_and_save_piece(island, piece_output_path):
            output_paths.append(piece_output_path)
            piece_id += 1

    print(f"> Extracted {len(output_paths)} pieces from {input_path.split('/')[-1]}")
    return output_paths


def _clean_and_save_piece(piece_coordinates, output_path):
    """
    Returns whether or not the piece was considered valid and was saved
    """

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
    img.save(output_path)
    return True

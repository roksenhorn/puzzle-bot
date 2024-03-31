import os
import PIL

from common import util


PIL.Image.MAX_IMAGE_PIXELS = 912340000


def extract_pieces(input_path, output_path, start_id):
    """
    Returns how many pieces were extracted
    """
    pixels, _, _ = util.binary_pixel_data_for_photo(input_path)

    def found_island(island, i):
        return clean_and_save_piece(i + start_id, island, output_path)

    islands = util.find_islands(pixels, callback=found_island, ignore_islands_along_border=True)
    print(f"Extracted {len(islands)} pieces from {input_path.split('/')[-1]}")
    return len(islands)


def clean_and_save_piece(piece_id, piece_coordinates, output_path):
    if len(piece_coordinates) < 100:
        return False

    # figure out the dimensions of the piece, then pad it with a small border
    BORDER_WIDTH_PX = 1
    xs = [x for (x, _) in piece_coordinates]
    ys = [y for (_, y) in piece_coordinates]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    width = (maxx - minx + 1) + (2 * BORDER_WIDTH_PX)
    height = (maxy - miny + 1) + (2 * BORDER_WIDTH_PX)

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

    img = PIL.Image.new('1', (width, height))
    img.putdata([pixel for row in pixels for pixel in row])
    img.save(os.path.join(output_path, f'{piece_id}.bmp'))
    return True

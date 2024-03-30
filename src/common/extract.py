import os
import PIL

from common import util


PIL.Image.MAX_IMAGE_PIXELS = 912340000


def extract_pieces(input_path, output_path=None):
    print(f"Loading {input_path.split('/')[-1]}...")
    pixels, _, _ = util.binary_pixel_data_for_photo(input_path)

    def save_island(island, i):
        return extract_piece(i + 1, island, output_path)

    print(f"Finding islands...")
    islands = util.find_islands(pixels, callback=save_island, ignore_islands_along_border=True)
    print(f"Found {len(islands)} pieces")


def extract_piece(piece_id, piece_coordinates, output_path=None):
    if len(piece_coordinates) < 100:
        return False

    BORDER_WIDTH_PX = 1
    xs = [x for (x, _) in piece_coordinates]
    ys = [y for (_, y) in piece_coordinates]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    width = (maxx - minx + 1) + (2 * BORDER_WIDTH_PX)
    height = (maxy - miny + 1) + (2 * BORDER_WIDTH_PX)
    print(f"{width} x {height}")

    if width < 0.3 * height or height < 0.3 * width:
        print(f"Skipping piece {piece_id} because it is too thin")
        return False

    pixels = []
    for i in range(height):
        pixels.append([])
        for j in range(width):
            pixels[i].append(0)

    for (x, y) in piece_coordinates:
        xx = x - minx + BORDER_WIDTH_PX
        yy = y - miny + BORDER_WIDTH_PX
        pixels[yy][xx] = 1

    if output_path:
        img = PIL.Image.new('1', (width, height))
        img.putdata([pixel for row in pixels for pixel in row])
        img.save(os.path.join(output_path, f'{piece_id}.bmp'))

    return True

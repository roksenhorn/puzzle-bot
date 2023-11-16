"""
Given a large bitmap image containing many pieces, extracts each piece into its own bitmap image
Processes the image by removing dust and hairs, then crops tightly leaving a 1px margin around the piece
"""
import argparse
import os
from PIL import Image

from common import util


def extract_pieces(input_path, n, output_path=None):
    print(f"Loading {input_path.split('/')[-1]}...")
    pixels, _, _ = util.load_binary_image(input_path)

    def save_island(island, i):
        return extract_piece(i + 1, island, output_path)

    print(f"Finding islands...")
    util.find_islands(pixels, callback=save_island)


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
        img = Image.new('1', (width, height))
        img.putdata([pixel for row in pixels for pixel in row])
        img.save(os.path.join(output_path, f'{piece_id}.bmp'))

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='Path to a binary bitmap image')
    parser.add_argument('--number-of-pieces', required=True, type=int, help='how many pieces are in the image')
    parser.add_argument('--output-path', required=False, default=None, help='directory to save the individual piece bitmaps to')
    args = parser.parse_args()

    extract_pieces(args.input_path, args.number_of_pieces, args.output_path)

"""
Given a large bitmap image containing many pieces, extracts each piece into its own bitmap image
Processes the image by removing dust and hairs, then crops tightly leaving a 1px margin around the piece
"""
import argparse
import PIL

from common import segment


def extract_pieces(input_path, n, output_path=None):
    with PIL.Image.open(input_path) as img:
        # Get image data as a 2D array of pixels
        width, height = img.size
        pixels = list(img.getdata())

    binary_pixels = []

    # Convert pixels to 0 or 1
    for i, pixel in enumerate(pixels):
        print(pixel, i)
        x = i % width
        y = i // width
        if y >= len(binary_pixels):
            binary_pixels.append([])
        binary_pixels[y].append(1 if pixel > 0 else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='Path to a binary bitmap image')
    parser.add_argument('--number-of-pieces', required=True, help='how many pieces are in the image')
    parser.add_argument('--output-path', required=False, default=None, help='directory to save the individual piece bitmaps to')
    args = parser.parse_args()

    extract_pieces(args.input_path, args.number_of_pieces, args.output_path)

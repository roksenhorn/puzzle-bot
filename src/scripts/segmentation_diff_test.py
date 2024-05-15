"""
A script to compare two segmentation methods and find the differences between them.
"""
import os
import numpy as np
import PIL
import re
import multiprocessing

from common import util


SEGMENT_DIR_A = '2_segmented_a'
SEGMENT_DIR_B = '2_segmented_b'


def run(path):
    # fill_islands(os.path.join(path, SEGMENT_DIR_A), os.path.join(path, SEGMENT_DIR_A))
    # fill_islands(os.path.join(path, SEGMENT_DIR_B), os.path.join(path, SEGMENT_DIR_B))

    for path_a in os.listdir(os.path.join(path, SEGMENT_DIR_A)):
        path_b = os.path.join(path, SEGMENT_DIR_B, path_a)
        path_a = os.path.join(path, SEGMENT_DIR_A, path_a)

        img_a, w_a, h_a = util.load_bmp_as_binary_pixels(path_a)
        img_b, w_b, h_b = util.load_bmp_as_binary_pixels(path_b)
        w = max(w_a, w_b)
        h = max(h_a, h_b)

        # Pad the smaller image with 0s using numpy
        img_a = np.pad(img_a, ((0, h - h_a), (0, w - w_a)))
        img_b = np.pad(img_b, ((0, h - h_b), (0, w - w_b)))

        ssd_score = util.normalized_ssd(img_a, img_b)
        if ssd_score > 3.0:
            print(f'{path_a}: {ssd_score}')


def fill_islands(input_path, output_path):
    fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(_fill_islands, args)


def _fill_islands(args):
    input_path, output_path = args
    pixels, width, height = util.load_bmp_as_binary_pixels(input_path)
    islands = _find_islands(pixels, ignore_islands_along_border=True, island_value=0)

    print(f"Found {len(islands)} islands in {input_path.split('/')[-1]}")
    for i, island in enumerate(islands):
        for (x, y) in island:
            pixels[y][x] = 1

    img = PIL.Image.new('1', (width, height))
    img.putdata([pixel for row in pixels for pixel in row])
    img.save(output_path)


def _find_islands(grid, callback=None, ignore_islands_along_border=False, island_value=1):
    """
    Given a grid of 0s and 1s, finds all "islands" of 1s:
    00000000
    01110000
    01111000
    00111110
    00000000

    :param grid: a 2D array of 0s and 1s
    :param callback: a function that will be called with each island found
    :param ignore_islands_along_border: if True, islands that touch the border of the grid will be ignored
    :param island_value: the value that represents an island in the grid (1 or 0)

    Returns either a list of islands, or a list of Trues if a callback was provided
    """
    visited1 = set()
    visited2 = set()
    islands = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == island_value and (i, j) not in visited1 and (i, j) not in visited2:
                island = set()
                queue = [(i, j)]
                touched_border = False

                # to prevent memory from getting too big and lookups from taking too long,
                # we maintain two visited sets, we check if we've visited a
                # location by checking either, and drain them offset
                if len(islands) % 160 == 0:
                    visited1 = set()
                if len(islands) % 160 == 80:
                    visited2 = set()
                while queue:
                    x, y = queue.pop(0)
                    if (x, y) not in visited1 and (x, y) not in visited2:
                        visited1.add((x, y))
                        visited2.add((x, y))
                        island.add((y, x))
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) and grid[x + dx][y + dy] == island_value:
                                queue.append((x + dx, y + dy))
                        if x == 0 or y == 0 or x == len(grid) - 1 or y == len(grid[0]) - 1:
                            touched_border = True

                if ignore_islands_along_border and touched_border:
                    continue

                if callback:
                    ok = callback(island, len(islands))
                    if ok:
                        islands.append(True)
                else:
                    islands.append(island)
    return islands


if __name__ == '__main__':
    run(path='data')
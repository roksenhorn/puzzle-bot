import os
from PIL import Image
import json
import numpy as np

from common import util, sides
import shutil


DUPLICATE_THRESHOLD = 1.0


def deduplicate(input_path, output_path):
    """
    Removes duplicate vector pieces by only copying over unique pieces to the output directory
    """

    # TODO:
    # Use the piece location to make sure we're only removing pieces that are physically at the same location

    # open up all the pieces
    i = 1
    pieces = {}
    while os.path.exists(os.path.join(input_path, f'side_{i}_0.json')):
        piece = []
        for j in range(4):
            with open(os.path.join(input_path, f'side_{i}_{j}.json')) as f:
                data = json.load(f)
                side = sides.Side(i, j, data['vertices'], piece_center=data['piece_center'], is_edge=data['is_edge'], resample=True)
                piece.append(side)
        pieces[i] = piece
        i += 1

    uniques = set()
    dupes = set()

    for i, sides0 in pieces.items():
        if i in dupes:
            continue

        found_dupe = False
        for j, sides1 in pieces.items():
            if i == j:
                continue

            score = _compare(sides0, sides1)
            if score < DUPLICATE_THRESHOLD:
                print(f"[{i}]\t Marking piece {j} as duplicate \t Similarity: {score}")
                dupes.add(j)
                found_dupe = True

        if not found_dupe:
            uniques.add(i)

    print(f"Started with {len(pieces)}; found {len(dupes)} duplicate pieces; resulting in {len(uniques)} unique pieces.")

    # finally, save off all the uniques
    for id in uniques:
        # copy the file to the output directory
        svg = f'{id + 1}.svg'
        shutil.copyfile(os.path.join(input_path, svg), os.path.join(output_path, svg))
        for i in range(4):
            side_i = f'side_{id}_{i}.json'
            shutil.copyfile(os.path.join(input_path, side_i), os.path.join(output_path, side_i))

    return len(uniques)


def _compare(sides0, sides1):
    """
    Compare this piece to another piece, returning a score of how similar they are
    0 = no error
    higher = more error
    """
    min_cumulative_error = None
    for start_i in range(4):
        cumulative_error = 0
        for p1_side_i in range(4):
            p0_side_i = (p1_side_i + start_i) % 4
            error = sides0[p0_side_i].error_when_fit_with(sides1[p1_side_i], flip=False, skip_edges=False, render=False)
            if error is None:
                cumulative_error = None
                break
            else:
                cumulative_error += error

        if min_cumulative_error is None or (cumulative_error is not None and cumulative_error < min_cumulative_error):
            min_cumulative_error = cumulative_error

    return min_cumulative_error
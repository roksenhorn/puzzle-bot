import os
import json
import numpy as np
import math
from glob import glob
import shutil

from common import util, sides



DUPLICATE_THRESHOLD = 1.5
SIDE_MISALIGNMENT_RAD = 15.0 * math.pi / 180


def deduplicate(input_path, output_path):
    """
    Removes duplicate vector pieces by only copying over unique pieces to the output directory
    """
    # TODO: use the piece location to make sure we're only removing pieces that are physically at the same location

    # open up all the pieces
    pieces = {}
    for path in glob(f"{input_path}/side_*_0.json"):
        i = int(path.split('/')[-1].split('_')[1])
        print(path)
        piece = []
        for j in range(4):
            with open(os.path.join(input_path, f'side_{i}_{j}.json')) as f:
                data = json.load(f)
                side = sides.Side(i, j, data['vertices'], piece_center=data['piece_center'], is_edge=data['is_edge'], resample=True, rotate=False)
                piece.append(side)
        pieces[i] = piece

    uniques = set()
    dupes = set()

    for i, sides0 in pieces.items():
        # if this piece is duplicating someone else, skip it
        if i in dupes:
            continue

        dupe_side_lens = {}
        dupe_side_lens[i] = sum([s.length for s in sides0])
        for j, sides1 in pieces.items():
            if i == j:
                continue

            score = _compare(sides0, sides1)
            if score < DUPLICATE_THRESHOLD:
                print(f"[{i}]\t is duplicated by {j} \t Similarity: {score}")
                dupe_side_lens[j] = sum([s.length for s in sides1])
            elif score < 5.0 * DUPLICATE_THRESHOLD:
                print(f"\t\t\t[{i}]\t is similar to {j} \t Similarity: {score}")

        if len(dupe_side_lens) == 1:
            # if this piece was truly unique, keep it
            uniques.add(i)
        else:
            # if this piece has duplciates, of all the duplicates, find the one with the cleanest vectorization
            sorted_side_lens = sorted(dupe_side_lens.items(), key=lambda x: (x[1], x[0]))
            best_dupe = sorted_side_lens[0][0]
            uniques.add(best_dupe)
            # print(f"\t Keeping {best_dupe} with len {sorted_side_lens[0][1]}")
            for j in sorted_side_lens[1:]:
                dupes.add(j[0])
                # print(f"\t Removing {j[0]} with len {j[1]}")

    print(f"Started with {len(pieces)}; found {len(dupes)} duplicate pieces; resulting in {len(uniques)} unique pieces.")

    # finally, copy all the uniques to the output directory
    for id in uniques:
        # copy the json files
        for i in range(4):
            side_i = f'side_{id}_{i}.json'
            shutil.copyfile(os.path.join(input_path, side_i), os.path.join(output_path, side_i))
        # copy the vector file as well
        vector_filename = glob(f"{id}_*.svg", root_dir=input_path)[0] # Take the 0th element, there should be exactly 1
        input_vector_file = os.path.join(input_path, vector_filename)
        output_vector_file = os.path.join(output_path, vector_filename)
        shutil.copyfile(input_vector_file, output_vector_file)

    return len(uniques)


def _compare(sides0, sides1):
    """
    Compare this piece to another piece, returning a score of how similar they are
    0 = no error
    higher = more error
    Note: we cannot assume that sides0[0] is the same as sides1[0] - they might be in different indices
    """

    side00, side01, side02, side03 = sides0
    permutations = [
        [side00, side01, side02, side03],
        [side01, side02, side03, side00],
        [side02, side03, side00, side01],
        [side03, side00, side01, side02]
    ]

    min_cumulative_error = 1000
    for sides0 in permutations:
        # first check to see if the pieces are in approximately the same orientation
        # we expect no rotation between duplicates
        sides_aligned = True
        for i in range(4):
            angle_diff = util.compare_angles(sides0[i].angle, sides1[i].angle)
            if angle_diff > SIDE_MISALIGNMENT_RAD:
                sides_aligned = False
                break
        if not sides_aligned:
            continue

        # if the sides are in approximately the same orientation, see how close to perfect matches they are
        cumulative_error = 0
        for i in range(4):
            cumulative_error += sides0[i].error_when_fit_with(sides1[i], flip=False, skip_edges=False, render=False)

        if cumulative_error < min_cumulative_error:
            min_cumulative_error = cumulative_error

    return min_cumulative_error
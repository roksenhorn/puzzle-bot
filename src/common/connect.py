import os
import json
from typing import List
import multiprocessing

from common import pieces, sides


# Building the graph took 440.38 seconds


SOLUTION = [
    # [(1, 0), (4, 2)],
    # [(1, 1), (2, 3)],
    # [(1, 2), None],
    # [(1, 3), (10, 2)],
    # [(3, 3), (2, 1)],
]


def build(input_path, output_path, id=None, serialize=False):
    print("> Loading piece data...")
    ps = pieces.Piece.load_all(input_path, resample=True)
    print("\t ...Loaded")

    if not serialize and id is None:
        pool = multiprocessing.Pool()
        with multiprocessing.Pool(processes=8) as pool:
            results = [pool.apply_async(_find_potential_matches_for_piece, (ps, piece_id)) for piece_id in ps.keys()]
            out = [r.get() for r in results]
    else:
        out = []
        for piece_id in ps.keys():
            if id is not None and piece_id != id:
                continue
            out.append(_find_potential_matches_for_piece(ps, piece_id, debug=(id is not None)))

    ps = { piece_id: piece for (piece_id, piece) in out }
    _save(ps, output_path)


def _find_potential_matches_for_piece(ps, piece_id, debug=False):
    """
    Find other sides that fit with this piece's sides
    """
    piece = ps[piece_id]

    # for all other piece's sides, find the ones that fit with this piece's sides
    for si, side in enumerate(piece.sides):
        if side.is_edge:
            continue

        for other_piece_id, other_piece in ps.items():
            if other_piece_id == piece_id:
                continue

            for sj, other_side in enumerate(other_piece.sides):
                if other_side.is_edge:
                    continue

                # for debugging, we can optionally provide side-matches from the actual solution and see how well the algo thinks they fit together
                part_of_solution = ([(piece_id, si), (other_piece_id, sj)] in SOLUTION) or ([(other_piece_id, sj), (piece_id, si)] in SOLUTION)

                # compute the error between our piece's side and this other piece's side
                error = side.error_when_fit_with(other_side, render=part_of_solution or debug, debug_str=f'{piece_id}[{si}] vs {other_piece_id}[{sj}]')
                if error <= sides.SIDE_MAX_ERROR_TO_MATCH:
                    piece.fits[si].append((other_piece.id, sj, error))

                if error > sides.SIDE_MAX_ERROR_TO_MATCH and part_of_solution:
                    raise ValueError(f"Should have matched but didn't: {piece_id}[{si}] vs {other_piece_id}[{sj}]")

        # make sure we have at least one match
        if len(piece.fits[si]) == 0:
            raise Exception(f'Piece {piece_id} side {si} has no matches but is not an edge')

        # sort by error
        piece.fits[si] = sorted(piece.fits[si], key=lambda x: x[2])
        least_error = piece.fits[si][0][2]

        # only keep the best matches
        piece.fits[si] = [f for f in piece.fits[si] if f[2] <= least_error * 3.0]

        print(f"Piece {piece_id}[{si}] has {len(piece.fits[si])} matches, best: {least_error}")
        if len(piece.fits[si]) > 5:
            fifth_match_error = piece.fits[si][4][2]
            print(f"\t1st match error: {least_error} \t ==> 5th match error: {fifth_match_error} \t ==> ratio: {fifth_match_error / least_error}")

    return (piece_id, piece)


def _save(pieces, out_directory):
    out = { p_id: p.to_dict() for (p_id, p) in pieces.items() }
    path = os.path.join(out_directory, 'connectivity.json')
    with open(path, 'w') as f:
        json.dump(out, f)

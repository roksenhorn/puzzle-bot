import os
import yaml
from typing import List, Tuple
import multiprocessing

from common import pieces, sides


SOLUTION = [
    [(1, 0), None],
    [(1, 1), (10, 0)],
    [(1, 2), (4, 0)],
    [(1, 3), (2, 1)],
    [(5, 2), (6, 1)],
    [(13, 0), (49, 2)],
    [(14, 2), (47, 3)],
    [(86, 0), (18, 0)],
]


def build(input_path, output_path, id=None):
    ps = pieces.Piece.load_all(input_path)
    if id is not None:
        ps = {id: ps[id]}

    pool = multiprocessing.Pool()
    with multiprocessing.Pool(processes=8) as pool:
        results = [pool.apply_async(_find_potential_matches_for_piece, (ps, piece_id)) for piece_id in ps.keys()]
        out = [r.get() for r in results]
        ps = { piece_id: piece for (piece_id, piece) in out }

    _save(ps, output_path)


def _find_potential_matches_for_piece(ps, piece_id):
    """
    Find other sides that fit with this piece's sides
    """
    piece = ps[piece_id]

    for other_piece_id, other_piece in ps.items():
        if other_piece_id == piece_id:
            continue

        for si, side in enumerate(piece.sides):
            if side.is_edge:
                continue

            for sj, other_side in enumerate(other_piece.sides):
                if other_side.is_edge:
                    continue

                part_of_solution = ([(piece_id, si), (other_piece_id, sj)] in SOLUTION) or ([(other_piece_id, sj), (piece_id, si)] in SOLUTION)
                error = side.error_when_fit_with(other_side, render=part_of_solution, debug_str=f'{piece_id}[{si}] vs {other_piece_id}[{sj}]')
                if error <= sides.SIDE_MAX_ERROR_TO_MATCH:
                    piece.fits[si].append((other_piece.id, sj, error))

                if error > sides.SIDE_MAX_ERROR_TO_MATCH and part_of_solution:
                    raise ValueError(f"Should have matched but didn't: {piece_id}[{si}] vs {other_piece_id}[{sj}]")

    for si, side in enumerate(piece.sides):
        if side.is_edge:
            continue

        if len(piece.fits[si]) == 0:
            raise Exception(f'Piece {piece_id} side {si} has no matches but is not an edge')

        # sort by error
        piece.fits[si] = sorted(piece.fits[si], key=lambda x: x[2])
        least_error = piece.fits[si][0][2]

        print(f"Piece {piece_id}[{si}] has {len(piece.fits[si])} matches, best: {least_error}")

    return (piece_id, piece)


def _save(pieces, out_directory):
    out = { p_id: p.to_dict() for (p_id, p) in pieces.items() }
    path = os.path.join(out_directory, 'connectivity.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, default_flow_style=True)

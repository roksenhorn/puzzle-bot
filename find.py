"""
Given a path to a photo and a path to the base puzzle dir, finds which piece this is
"""
import time
import os
import yaml
import argparse

import segment
import vector
import sides


VECTOR_DIR = '2_vector'


def find(photo_path, puzzle_dir):
    # vectorize the input photo
    pixels, w, h = segment.segment(photo_path)
    print(w, 'x', h)
    v = vector.Vector(pixels, w, h, id=0).process(render=True)

    # open up all the pieces
    print("Loading all piece data...")
    i = 1
    pieces = {}
    while os.path.exists(os.path.join(puzzle_dir, VECTOR_DIR, f'side_{i}_0.yaml')):
        piece = []
        for j in range(4):
            with open(os.path.join(puzzle_dir, VECTOR_DIR, f'side_{i}_{j}.yaml')) as f:
                data = yaml.safe_load(f)
                side = sides.Side(i, j, data['vertices'], piece_center=data['piece_center'], is_edge=data['is_edge'])
                piece.append(side)
        pieces[i] = piece
        i += 1

    print(f"Loaded {i-1} pieces")

    scores = {}
    best_score = None
    best_i = None
    for i, piece in pieces.items():
        scores[i] = v.compare(piece)
        print(f"Comparing to {i} ==> {scores[i]:.2f}")
        if best_score is None or scores[i] < best_score:
            best_score = scores[i]
            best_i = i

    print(f"Best score: {best_score} ==> (piece {best_i})")
    return best_i


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find a piece in a puzzle')
    parser.add_argument('--photo-path', type=str, required=True, help='Path to the photo')
    parser.add_argument('--puzzle-dir', type=str,  required=True, help='Path to the puzzle directory')
    args = parser.parse_args()

    find(args.photo_path, args.puzzle_dir)
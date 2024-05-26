"""
Given a path to processed piece data, finds a solution
"""

import os
import time
import json
import math

from common import board, connect, dedupe, util
from common.config import *


def solve(path, start_at=3):
    """
    Given a path to processed piece data, finds a solution
    """
    if start_at <= 3:
        _deduplicate(batch_data_path=os.path.join(path, PHOTOS_DIR, "batch.json"), input_path=os.path.join(path, VECTOR_DIR), output_path=os.path.join(path, DEDUPED_DIR))
    if start_at <= 4:
        connectivity = _find_connectivity(input_path=os.path.join(path, DEDUPED_DIR), output_path=os.path.join(path, CONNECTIVITY_DIR))
    else:
        connectivity = None

    puzzle = _build_board(connectivity=connectivity, input_path=os.path.join(path, CONNECTIVITY_DIR), output_path=os.path.join(path, SOLUTION_DIR), metadata_path=os.path.join(path, VECTOR_DIR))
    _move_pieces_into_place(puzzle=puzzle, metadata_path=os.path.join(path, VECTOR_DIR), output_path=os.path.join(path, SOLUTION_DIR))


def _deduplicate(batch_data_path, input_path, output_path):
    """
    Often times the same piece was successfully extracted from multiple photos
    We do this on vectorized pieces to ignore noise in BMPs
    """
    print(f"\n{util.RED}### 3 - Deduplicating vector pieces ###{util.WHITE}\n")
    count = dedupe.deduplicate(batch_data_path, input_path, output_path)
    if count != PUZZLE_NUM_PIECES:
        raise Exception(f"Expected {PUZZLE_NUM_PIECES} pieces, but found {count}")


def _find_connectivity(input_path, output_path):
    """
    Opens each piece data and finds how each piece could connect to others
    """
    print(f"\n{util.RED}### 4 - Building connectivity ###{util.WHITE}\n")
    start_time = time.time()
    connectivity = connect.build(input_path, output_path)
    duration = time.time() - start_time
    print(f"Building the graph took {round(duration, 2)} seconds")
    return connectivity


def _build_board(connectivity, input_path, output_path, metadata_path):
    """
    Searches connectivity to find the solution
    """
    print(f"\n{util.RED}### 5 - Finding where each piece goes ###{util.WHITE}\n")
    start_time = time.time()
    puzzle = board.build(connectivity=connectivity, input_path=input_path, output_path=output_path)
    duration = time.time() - start_time
    print(f"Finding where each piece goes took {round(duration, 2)} seconds")
    return puzzle


def _move_pieces_into_place(puzzle, metadata_path, output_path):
    """
    Compute how each piece must be moved from its original photo space to the final board location

    We first align border pieces with a virtual edge to the solution is bounded to a perfect rectangle
    As we place pieces, we rotate and translate them to fit snuggly against their neighbors

    We work our way clockwise around the border, then spiral inward

    For a given peice, the the orientation of that piece's sides walks clockwise:

            (0)
        ----Top--->
       ^           |
       |          Right (1)
    Left (3)       |
       |           V
       <----Bot----
            (2)
    """
    # We'll store how each piece is rotated (around its incenter) then translated
    outputs = {}
    viz_data = []
    placed_pieces = {}

    # as we spiral around, we make sure each ring of the spiral is a perfect rectangle
    spiral_width, spiral_height = (0, 0)

    # We'll place pieces by wrapping around the puzzle's border, starting at the top left corner
    # and rotating clockwise around the border, then spiraling inward
    x, y = (0, 0)
    direction = (+1, 0) # we start by moving right (+1 in the x)
    for i in range(puzzle.width * puzzle.height):
        piece_id, _, orientation = puzzle.get(x, y)
        print(f"> Placing Piece {piece_id} in spot [{x}, {y}], in orientation {orientation}")

        # if we're placing a border piece, we create a virtual piece beyond the border for perfect rectangular alignment
        if y == 0:  # top border: create a fake bottom side above
            placed_pieces[(x, y - 1)] = [[], [], [(100, 0), (0, 0)], []]  # `Bottom` side points right
        if x == 0:  # left border: create a fake right side to the left
            placed_pieces[(x - 1, y)] = [[], [(0, 0), (0, 100)], [], []]  # `Right` side points down
        if x == puzzle.width - 1:  # right border: create a fake left side to the right
            placed_pieces[(x + 1, y)] = [[], [], [], [(spiral_width, 100), (spiral_width, 0)]]  # `Left` side points up
        if y == puzzle.height - 1:  # bottom border: create a fake top side below
            placed_pieces[(x, y + 1)] = [[(0, spiral_height), (100, spiral_height)], [], [], []]  # `Top` side points left

        # Extract relevant sides from any neighbors that have already been placed (including fake pieces for the border)
        neighbor_above = placed_pieces.get((x, y - 1), [[], [], [], []])[2]  # grab the bottom side of the neighbor above us
        neighbor_right = placed_pieces.get((x + 1, y), [[], [], [], []])[3]  # grab the left side of the neighbor to our right
        neighbor_below = placed_pieces.get((x, y + 1), [[], [], [], []])[0]  # grab the top side of the neighbor below us
        neighbor_left = placed_pieces.get((x - 1, y), [[], [], [], []])[1]  # grab the right side of the neighbor to our left

        # compute the orientation of the neighbor so we know how we need to be oriented to plug into that neighbor
        neighbor_above_angle = util.angle_between(neighbor_above[0], neighbor_above[-1]) % (2 * math.pi) if neighbor_above else None
        neighbor_right_angle = util.angle_between(neighbor_right[0], neighbor_right[-1]) % (2 * math.pi) if neighbor_right else None
        neighbor_below_angle = util.angle_between(neighbor_below[0], neighbor_below[-1]) % (2 * math.pi) if neighbor_below else None
        neighbor_left_angle = util.angle_between(neighbor_left[0], neighbor_left[-1]) % (2 * math.pi) if neighbor_left else None

        # load our piece's side data
        sides = []
        for i in range(4):
            with open(os.path.join(metadata_path, f'side_{piece_id}_{i}.json'), 'r') as f:
                sides.append(json.load(f))

        # what angle is each side currently at?
        side_angles = []
        for side in sides:
            vertices = side['vertices']
            angle = util.angle_between(vertices[0], vertices[-1]) % (2 * math.pi)
            side_angles.append(angle)

        # we need to rotate the piece such that side 0 ends up in `orientation`, and all the other sides are rotated accordingly
        # `new_top`` is the index into `sides` that we want to be the top side after rotation, connecting to the neighbor above's bottom
        if orientation == 0:
            # we're in the correct orientation
            new_top = 0
            new_right = 1
            new_bottom = 2
            new_left = 3
        elif orientation == 1:
            # our top is pointing to the left (side 3), so we need to rotate the piece 90° clockwise
            new_top = 3
            new_right = 0
            new_bottom = 1
            new_left = 2
        elif orientation == 2:
            # our top is on the bottom (side 2), so we need to rotate the piece 180°
            new_top = 2
            new_right = 3
            new_bottom = 0
            new_left = 1
        elif orientation == 3:
            # our top is pointing to the right (side 1), so we need to rotate the piece 90° counterclockwise
            new_top = 1
            new_right = 2
            new_bottom = 3
            new_left = 0

        # compute how much each side wants to rotate to oppose their neighbors, and align to one of the neighbors
        top_side_rotation = (neighbor_above_angle - side_angles[new_top] - math.pi) if neighbor_above else None
        right_side_rotation = (neighbor_right_angle - side_angles[new_right] - math.pi) if neighbor_right else None
        bottom_side_rotation = (neighbor_below_angle - side_angles[new_bottom] - math.pi) if neighbor_below else None
        left_side_rotation = (neighbor_left_angle - side_angles[new_left] - math.pi) if neighbor_left else None
        if direction == (1, 0):
            rotation = top_side_rotation
        elif direction == (0, 1):
            rotation = right_side_rotation
        elif direction == (-1, 0):
            rotation = bottom_side_rotation
        elif direction == (0, -1):
            rotation = left_side_rotation

        # actually rotate the sides, around our incenter
        rotated_sides = []
        for side in sides:
            rotated_side = util.rotate_polyline(side['vertices'], around_point=side["incenter"], angle=rotation)
            rotated_sides.append(rotated_side)

        # Now we'll compute how much to translate
        # but first, for border pieces, we need to know where the virtual neighbors would be
        # we previously made fake neighbor sides for the border to get the angle right
        # let's update the fake neighbors to match the actual positions these pieces should go
        if y == 0:  # top border
            origin_x = neighbor_left[0][0]
            origin_y = 0  # force top border to be at y=0
            w = rotated_sides[new_top][-1][0] - rotated_sides[new_top][0][0]
            neighbor_above = [(origin_x + w, origin_y), (origin_x, origin_y)]
        if x == puzzle.width - 1:  # right border
            if y == 0:
                # when we've made it to the top right corner, figure out how wide the puzzle is
                piece_width = rotated_sides[new_top][-1][0] - rotated_sides[new_top][0][0]
                spiral_width = neighbor_left[0][0] + piece_width
                print(f"Setting spiral_width to {spiral_width}")

            origin_x = spiral_width  # force right border to be against x=spiral_width
            origin_y = neighbor_above[0][1]
            h = rotated_sides[new_right][-1][1] - rotated_sides[new_right][0][1]
            neighbor_right = [(origin_x, origin_y + h), (origin_x, origin_y)]
        if y == puzzle.height - 1:  # bottom border
            if x == puzzle.width - 1:
                # when we've made it to the bottom right corner, figure out how tall the puzzle is
                piece_height = rotated_sides[new_right][-1][1] - rotated_sides[new_right][0][1]
                spiral_height = neighbor_above[0][1] + piece_height
                print(f"Setting spiral_height to {spiral_height}")

            origin_x = neighbor_right[0][0]
            origin_y = spiral_height  # force bottom border to be against y=spiral_height
            w = rotated_sides[new_bottom][-1][0] - rotated_sides[new_bottom][0][0]
            neighbor_below = [(origin_x - w, origin_y), (origin_x, origin_y)]
        if x == 0 and y != 0:  # left border, ignoring top left corner
            if y == puzzle.height - 1:
                # bottom left corner needs to x-align with 0 and y-align with the height
                origin_x = 0
                origin_y = spiral_height
                w = rotated_sides[new_bottom][0][0] - rotated_sides[new_bottom][-1][0]
            else:
                origin_x = 0
                origin_y = neighbor_below[0][1]
            h = rotated_sides[new_left][-1][1] - rotated_sides[new_left][0][1]
            neighbor_left = [(origin_x, origin_y - h), (origin_x, origin_y)]

        # to get the best alignment, we want to average how much we'd need to translate to each of our existing neighbors
        samples = []
        if neighbor_above:
            print(f"\t > Neighbor Above:")
            samples.append(util.subtract(neighbor_above[-1], rotated_sides[new_top][0]))
        if neighbor_right:
            print(f"\t > Neighbor Right:")
            samples.append(util.subtract(neighbor_right[-1], rotated_sides[new_right][0]))
        if neighbor_below:
            print(f"\t > Neighbor Below:")
            samples.append(util.subtract(neighbor_below[-1], rotated_sides[new_bottom][0]))
        if neighbor_left:
            print(f"\t > Neighbor Left:")
            samples.append(util.subtract(neighbor_left[-1], rotated_sides[new_left][0]))

        for sample in samples:
            print(f"\t > Sample: {sample}")

        if x == 0 and y == 0:
            # make sure the first piece actually ends up at (0, 0)
            translation = util.subtract((0, 0), rotated_sides[new_top][0])
        else:
            # all other pieces will be translated by the average of how much they need to move to connect to each neighbor
            translation = util.multimidpoint(samples)

        # perform the translation
        incenter = (sides[0]["incenter"][0] + translation[0], sides[0]["incenter"][1] + translation[1])
        translated_rotated_sides = [util.translate_polyline(side, translation) for side in rotated_sides]

        # place the piece so future pieces have neighbors to grab onto
        placed_pieces[(x, y)] = [
            translated_rotated_sides[new_top],
            translated_rotated_sides[new_right],
            translated_rotated_sides[new_bottom],
            translated_rotated_sides[new_left]
        ]
        outputs[piece_id] = {
            "photo_space_origin": sides[0]["photo_space_origin"],
            "photo_space_incenter": sides[0]["photo_space_incenter"],
            "robot_state": sides[0]["robot_state"],
            "dest_photo_space_incenter": incenter,
            "dest_rotation": rotation,
            "solution_x": x,
            "solution_y": y,
        }
        print(f"\t > Rotate by {round(rotation * 180 / math.pi, 1)}° and translate by {translation}")

        # save off data for visualization purposes
        viz_data.append({"vertices": translated_rotated_sides[new_top], "is_edge": y == 0, "incenter": incenter})
        viz_data.append({"vertices": translated_rotated_sides[new_right], "is_edge": x == puzzle.width - 1, "incenter": incenter})
        viz_data.append({"vertices": translated_rotated_sides[new_bottom], "is_edge": y == puzzle.height - 1, "incenter": incenter})
        viz_data.append({"vertices": translated_rotated_sides[new_left], "is_edge": x == 0, "incenter": incenter})

        # when we hit the bounds or a piece we've placed, spiral around the boarder, inward
        next_x, next_y = x + direction[0], y + direction[1]
        if next_x < 0 or next_x >= puzzle.width or next_y < 0 or next_y >= puzzle.height or placed_pieces.get((next_x, next_y)):
            if direction == (+1, 0):
                # when we complete the top edge, we descend down the right edge
                direction = (0, +1)
                print("==================================")
                print(f" ----> TURNING DOWN ----v    spiral_width: {spiral_width}")
            elif direction == (0, +1):
                # when we complete the right edge, we move left along the bottom edge
                direction = (-1, 0)
            elif direction == (-1, 0):
                # when we complete the bottom edge, we move up the left edge
                direction = (0, -1)
            elif direction == (0, -1):
                # when we complete the left edge, we move right along the top edge
                direction = (+1, 0)
        x, y = (x + direction[0], y + direction[1])

    # generate a giant debug SVG of the final board
    svg = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    svg += f'<svg width="5000" height="4000" viewBox="-10 -10 5020 4020" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
    colors = ['cc0000', '999900', '00aa99', '3300bb']
    for i, side in enumerate(viz_data):
        pts = ' '.join([','.join([str(e / 5.0) for e in v]) for v in side["vertices"]])
        stroke_width = 1.5 if side["is_edge"] else 1.0
        dash = 'stroke-dasharray="9,3"' if side["is_edge"] else ''
        svg += f'<polyline points="{pts}" style="fill:none; stroke:#{colors[i % len(colors)]}; stroke-width:{stroke_width}" {dash} />'
        if i % 4 == 0:
            svg += f'<circle cx="{side["incenter"][0] / 5.0}" cy="{side["incenter"][1] / 5.0}" r="{1.0}" style="fill:#bb4400; stroke-width:0" />'
    svg += '</svg>'
    with open(os.path.join(output_path, "board.svg"), 'w') as f:
        f.write(svg)

    for piece_id, output in outputs.items():
        piece_output_path = os.path.join(output_path, f'{piece_id}.json')
        with open(piece_output_path, 'w') as f:
            json.dump(output, f)

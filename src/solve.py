"""
Given a path to processed piece data, finds a solution
"""

import os
import time
import json
import math

from common import board, connect, dedupe, util
from common.config import *


def solve(path):
    """
    Given a path to processed piece data, finds a solution
    """
    _deduplicate(input_path=os.path.join(path, VECTOR_DIR), output_path=os.path.join(path, DEDUPED_DIR))
    connectivity = _find_connectivity(input_path=os.path.join(path, DEDUPED_DIR), output_path=os.path.join(path, CONNECTIVITY_DIR))
    _build_board(connectivity=connectivity, input_path=os.path.join(path, CONNECTIVITY_DIR), output_path=os.path.join(path, SOLUTION_DIR), metadata_path=os.path.join(path, VECTOR_DIR))


def _deduplicate(input_path, output_path):
    """
    Often times the same piece was successfully extracted from multiple photos
    We do this on vectorized pieces to ignore noise in BMPs
    """
    print(f"\n{util.RED}### 3 - Deduplicating vector pieces ###{util.WHITE}\n")
    count = dedupe.deduplicate(input_path, output_path)
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

    # Now we must compute how each piece must be moved, from its original photo space to the final board location
    # We start at the top left corner:
    # We define the origin of the solution space to be the top left corner of that piece
    # and we rotate that piece to be orthogonal with the solution space axes (+x to the right, +y down)
    # we then place the next piece to the right of the first piece,
    # translating and rotating it so it lines up with the first piece's right edge
    # the orientation of sides walks clockwise:
    #
    #     ----T--->
    #    ^         |
    #    |         R
    #    L         |
    #    |         V
    #     <---B----
    rotations = {}
    incenters = {}

    debug_sides = []

    prior_row_bottoms = [[(100, 0), (0, 0)]] * puzzle.width
    print(prior_row_bottoms)

    for y in range(puzzle.height):
        if y > 150:
            break

        # at the start of each row, we reset the left neighbor to a virtual piece that's just a straight line down
        neighbor_above = prior_row_bottoms[0]
        h = neighbor_above[-1][1]  # this is side B, so grab the last vertex's (left-most) y coordinate
        neighbor_left = [(0, h), (0, h + 100)]  # vertical pointing down (side R)
        this_row_bottoms = []
        for x in range(puzzle.width):
            neighbor_above = prior_row_bottoms[x]
            neighbor_above_angle = util.angle_between(neighbor_above[0], neighbor_above[-1]) % (2 * math.pi)
            neighbor_left_angle = util.angle_between(neighbor_left[0], neighbor_left[-1]) % (2 * math.pi)

            piece_id, _, orientation = puzzle.get(x, y)
            print(f"\n=============\n[{x}, {y}] {piece_id} @ {orientation}")

            # open the piece's metadata
            sides = []
            for i in range(4):
                with open(os.path.join(metadata_path, f'side_{piece_id}_{i}.json'), 'r') as f:
                    sides.append(json.load(f))

            side_angles = []
            for side in sides:
                vertices = side['vertices']
                angle = util.angle_between(vertices[0], vertices[-1]) % (2 * math.pi)
                side_angles.append(angle)

            # rotate the piece such that side 0 ends up in `orientation`, and all the other sides are rotated accordingly
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

            # compute how much each side wants to rotate to oppose their neighbors
            # if we're along the top edge, we make sure we align to the top neighbor
            # if we're along the left edge, we make sure we align to the left neighbor
            # this helps prevent error from stacking up as we solve across the board
            top_side_rotation = (neighbor_above_angle - side_angles[new_top] - math.pi)
            left_side_rotation = (neighbor_left_angle - side_angles[new_left] - math.pi)
            if y == 0:
                rotation = top_side_rotation
            elif x == 0:
                rotation = left_side_rotation
            elif x == puzzle.width - 1:
                # at the end of the row, let's make sure the last piece's right wall is vertical
                angle_of_new_right_side = side_angles[new_right]
                rotation = -angle_of_new_right_side + math.pi / 2
            else:
                rotation = util.average_angles(top_side_rotation, left_side_rotation)


            print(f"Neighbor above @ {neighbor_above_angle * 180 / math.pi}°")
            print(f"Neighbor to left @ {neighbor_left_angle * 180 / math.pi}°")
            print(f"New top = side {new_top} @ {side_angles[new_top] * 180 / math.pi}°, should oppose {neighbor_above_angle * 180 / math.pi}° => rotate it {top_side_rotation * 180 / math.pi}°")
            print(f"New left = side {new_left} @ {side_angles[new_left] * 180 / math.pi}°, should oppose {neighbor_left_angle * 180 / math.pi}° => rotate it {left_side_rotation * 180 / math.pi}°")
            mismatch = util.compare_angles(top_side_rotation, left_side_rotation)
            # if mismatch > 8 * math.pi / 180:
            #     raise Exception(f"Rotation mismatch: {mismatch * 180 / math.pi} > 8°: {top_side_rotation * 180 / math.pi}° vs {left_side_rotation * 180 / math.pi}°")

            rotated_sides = []
            for side in sides:
                rotated_side = util.rotate_polyline(side['vertices'], around_point=side["incenter"], angle=rotation)
                new_side = {"vertices": rotated_side, "is_edge": side["is_edge"]}
                rotated_sides.append(new_side)
                debug_sides.append(new_side)

            # translate the piece to the correct position by pulling the intersection point of the two neighbors
            neighbor_left_origin = neighbor_left[0]
            if y > 0:
                neighbor_above_origin = neighbor_above[-1]
                start_origin = ((neighbor_left_origin[0] + neighbor_above_origin[0]) / 2, (neighbor_left_origin[1] + neighbor_above_origin[1]) / 2)
                end_origin = neighbor_above[0]
                start_translation = (start_origin[0] - rotated_sides[new_left]["vertices"][-1][0], start_origin[1] - rotated_sides[new_left]["vertices"][-1][1])
                end_translation = (end_origin[0] - rotated_sides[new_top]["vertices"][-1][0], end_origin[1] - rotated_sides[new_top]["vertices"][-1][1])
                translation = ((start_translation[0] + end_translation[0]) / 2, (start_translation[1] + end_translation[1]) / 2)
            else:
                # for the first row, we don't have real neighbors above us
                translation = (neighbor_left_origin[0] - rotated_sides[new_left]["vertices"][-1][0], neighbor_left_origin[1] - rotated_sides[new_left]["vertices"][-1][1])
            incenter = (sides[0]["incenter"][0] + translation[0], sides[0]["incenter"][1] + translation[1])
            print(f"\nNeighbor left's start: {neighbor_left[0]}")
            print(f"Piece's left's start: {rotated_sides[new_left]['vertices'][0]}")
            print(f"incenter from {sides[0]['incenter']} to {incenter} by {translation}")
            print(f"Rotating piece {piece_id} by {rotation * 180 / math.pi}° and translating by {translation}")
            for side in rotated_sides:
                side["vertices"] = util.translate_polyline(side["vertices"], translation)
                side["incenter"] = incenter

            rotations[piece_id] = rotation
            incenters[piece_id] = incenter

            neighbor_left = rotated_sides[new_right]["vertices"]
            this_row_bottoms.append(rotated_sides[new_bottom]["vertices"])

        prior_row_bottoms = this_row_bottoms

    # generate a giant debug SVG of the final board
    svg = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    svg += f'<svg width="3000" height="3000" viewBox="-10 -10 3020 3020" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
    colors = ['cc0000', '999900', '00aa99', '3300bb']
    for i, side in enumerate(debug_sides):
        pts = ' '.join([','.join([str(e / 5.0) for e in v]) for v in side["vertices"]])
        stroke_width = 1.5 if side["is_edge"] else 1.0
        dash = 'stroke-dasharray="9,3"' if side["is_edge"] else ''
        svg += f'<polyline points="{pts}" style="fill:none; stroke:#{colors[i % len(colors)]}; stroke-width:{stroke_width}" {dash} />'
        if i % 4 == 0:
            svg += f'<circle cx="{side["incenter"][0] / 5.0}" cy="{side["incenter"][1] / 5.0}" r="{1.0}" style="fill:#bb4400; stroke-width:0" />'
    svg += '</svg>'
    with open(output_path + "/board.svg", 'w') as f:
        f.write(svg)

    for i in range(1, 1001):
        for j in range(4):
            metadata_input_path = os.path.join(metadata_path, f'side_{i}_{j}.json')
            solution_output_path = os.path.join(output_path, f'side_{i}_{j}.json')
            with open(metadata_input_path, 'r') as f:
                metadata = json.load(f)
                metadata['dest_rotation'] = rotations[i-1]
                metadata['dest_photo_space_incenter'] = incenters[i-1]
            with open(solution_output_path, 'w') as f:
                json.dump(metadata, f)


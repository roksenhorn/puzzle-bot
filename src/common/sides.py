import math
from typing import List, Tuple
import numpy as np

from common import util


# Two sides from different pieces "fit" if they are within this threshold (0.0 = perfect)
SIDE_MAX_ERROR_TO_MATCH = 3.5  # 1.80

# sides must be within this multiple of each other's polyline length
SIDE_MAX_LENGTH_DISCREPANCY = 0.08

# when we resample a side, we use this many vertices
SIDE_RESAMPLE_VERTEX_COUNT = 26


class Side(object):
    def __init__(self, piece_id, side_id, vertices, piece_center, is_edge, resample=False, rotate=True, photo_filename=None) -> None:
        self.piece_id = piece_id
        self.side_id = side_id
        self.piece_center = piece_center
        self.is_edge = is_edge
        self.vertices = vertices
        self.p1 = vertices[0]
        self.p2 = vertices[-1]
        self.photo_filename = photo_filename

        if resample:
            vertices, self.v_length = util.resample_polyline(vertices, n=SIDE_RESAMPLE_VERTEX_COUNT)
            if rotate:
                angle = self.angle
                self.vertices = Side.rotated(vertices=vertices, from_angle=angle, desired_angle=0)  # aligned to be horizontal
                self.vertices_flipped = Side.rotated(vertices=vertices, from_angle=angle, desired_angle=math.pi)[::-1] # aligned to be horizontal but rotated and mirrored to be the negative space of the side
            else:
                self.vertices = np.array(vertices)
            self.p1 = self.vertices[0]
            self.p2 = self.vertices[-1]
        else:
            self.v_length = util.polyline_length(vertices)

    def __repr__(self) -> str:
        return f"Side({self.p1}->{self.p2} @ {int(self.angle * 180/math.pi)} deg, len={self.length}, n_vertices={len((self.vertices))}, is_edge={self.is_edge})"

    @property
    def angle(self) -> float:
        angle = util.angle_between(self.p1, self.p2)
        return angle

    @property
    def segment(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self.p1, self.p2)

    @property
    def length(self) -> float:
        return util.distance(self.p1, self.p2)

    def error_when_fit_with(self, side, flip=True, render=False, skip_edges = True, debug_str=None) -> bool:
        """
        Returns None if no match, or a float representing the similarity of the two sides (1.0 = perfect) if they generally match
        """
        # if render and debug_str:
        #     print(debug_str)

        if skip_edges and (self.is_edge or side.is_edge):
            # if render:
            #     print("\tNO MATCH: one is an edge!!!!!!!!!!")
            return 1000

        # sides must be roughly the same length
        d_scale = 1.0 - (self.length / side.length)
        if abs(d_scale) > SIDE_MAX_LENGTH_DISCREPANCY:
            # if render:
            #     print(f"\tNO MATCH: scale is too different!!!!!!!!!! {d_scale}")
            return 1000

        polyline1 = self.vertices
        if flip:  # plugging one piece into another means we need them to be inverse shapes
            polyline2 = side.vertices_flipped
        else:  # comparing two sides to see if they belong to the same piece
            polyline2 = side.vertices

        error, shift = util.error_between_polylines(polyline1, polyline2, p1_len=side.v_length)

        if render and debug_str and error <= SIDE_MAX_ERROR_TO_MATCH:
            print(debug_str)
            shifted0 = [(x - shift[0], y - shift[1]) for x, y in polyline1]
            print(f"\t ==> Error = {error}, shift: {shift}")
            util.render_polylines([shifted0, polyline2])

        return error

    @staticmethod
    def rotated(vertices, from_angle, desired_angle) -> List[Tuple[int, int]]:
        """
        Returns a list of vertices that have been geometrically rotated such that the side is at the desired angle, with p1 as the origin
        """
        o = vertices[0]

        # translate to origin
        translated = []
        for i, (x, y) in enumerate(vertices):
            translated.append((x - o[0], y - o[1]))

        angle_diff = desired_angle - from_angle
        rotated = []

        # rotate around the origin
        for v in translated:
            rotated.append(util.rotate(v, around=translated[0], angle=angle_diff))

        if desired_angle != 0:
            min_x = min([v[0] for v in rotated])
            rotated = [(v[0] - min_x, v[1]) for v in rotated]

        return np.array(rotated)

import math
from typing import List, Tuple

from common import util

# Two sides from different pieces "fit" if they are within this threshold (1.0 = perfect)
SIDE_MAX_ERROR_TO_MATCH = 0.6


class Side(object):
    def __init__(self, piece_id, side_id, vertices, piece_center, is_edge) -> None:
        self.piece_id = piece_id
        self.side_id = side_id

        self.vertices = vertices
        self.p1 = vertices[0]
        self.p2 = vertices[-1]
        self.piece_center = piece_center
        self.is_edge = is_edge

    def __repr__(self) -> str:
        return f"Side({self.p1}->{self.p2} @ {int(self.angle * 180/math.pi)} deg, len={self.length}, n_vertices={len((self.vertices))}, is_edge={self.is_edge})"

    def recompute_endpoints(self):
        self.p1 = self.vertices[0]
        self.p2 = self.vertices[-1]

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

    def error_when_fit_with(self, side, flip=True, render=False, debug_str=None) -> bool:
        """
        Returns None if no match, or a float representing the similarity of the two sides (1.0 = perfect) if they generally match
        """
        if render and debug_str:
            print(debug_str)

        if self.is_edge or side.is_edge:
            if render:
                print("\tNO MATCH: one is an edge!!!!!!!!!!")
            return 1000

        # sides must be roughly the same length
        scale = self.length / side.length
        if scale < 0.92 or scale > 1.08:
            if render:
                print(f"\tNO MATCH: scale is too different!!!!!!!!!! {scale}")
            return 1000

        polyline0 = self.rotated()
        if flip:  # plugging one piece into another means we need them to be inverse shapes
            polyline1 = side.rotated(math.pi)[::-1]
        else:  # comparing two sides to see if they belong to the same piece
            polyline1 = side.rotated()

        try:
            error, shift = util.error_between_polylines(polyline0, polyline1)
        except Exception as e:
            print(side.rotated(math.pi)[::-1])
            raise e

        if render and debug_str:
            shifted0 = [(x - shift[0], y - shift[1]) for x, y in polyline0]
            print(f"\t ==> Error = {error}, shift: {shift}")
            util.render_polylines([polyline0, polyline1])
            util.render_polylines([shifted0, polyline1])

        return error

    def rotated(self, desired_angle=0) -> List[Tuple[int, int]]:
        """
        Returns a list of vertices that have been geometrically rotated such that the side is at the desired angle, with p1 as the origin
        """

        # translate to origin
        translated = []
        for i, (x, y) in enumerate(self.vertices):
            translated.append((x - self.p1[0], y - self.p1[1]))

        angle_diff = desired_angle - self.angle
        rotated = []

        # rotate around the origin
        for v in translated:
            rotated.append(util.rotate(v, around=translated[0], angle=angle_diff))

        if desired_angle != 0:
            min_x = min([v[0] for v in rotated])
            rotated = [(v[0] - min_x, v[1]) for v in rotated]

        return rotated
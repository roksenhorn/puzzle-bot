import math
import os
import yaml
from PIL import Image
from typing import List, Tuple
import numpy as np

from common import sides, util


# How much error to tolerate when simplifying the shape
SIMPLIFY_EPSILON = 1.5

# We'll merge vertices closer than this
MERGE_IF_CLOSER_THAN_PX = 1.75

# Opposing sides must be "parallel" within this threshold (in degrees)
SIDE_PARALLEL_THRESHOLD_DEG = 32

# Adjacent sides must be "orthogonal" within this threshold (in degrees)
SIDES_ORTHOGONAL_THRESHOLD_DEG = 30

# A side must be at least this long to be considered an edge
EDGE_WIDTH_MIN_RATIO = 0.4


class Vector(object):
    @staticmethod
    def from_file(filename, id) -> 'Vector':
        # Open image file
        binary_pixels, width, height = util.load_binary_image(filename)
        v = Vector(pixels=binary_pixels, width=width, height=height, id=id)
        return v

    def __init__(self, pixels, width, height, id) -> None:
        self.pixels = pixels
        self.width = width
        self.height = height
        self.dim = float(self.width + self.height) / 2.0
        self.id = id
        self.sides = []

    def process(self, output_path=None, render=False):
        self.find_border_raster()
        self.vectorize()
        self.simplify()

        try:
            self.find_four_corners()
            self.insert_corners()
            self.find_four_sides()
        except Exception as e:
            self.render()
            raise e

        if render:
            self.render()

        if output_path:
            self.save(output_path)
        else:
            return self

    def save(self, output_path) -> None:
        full_svg_path = os.path.join(output_path, f"{self.id}_full.svg")
        colors = ['cc0000', '999900', '00aa99', '3300bb']
        svg = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
        svg += f'<svg width="{3 * self.width}" height="{3 * self.height}" viewBox="-10 -10 {20 + self.width} {20 + self.height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
        for i, side in enumerate(self.sides):
            pts = ' '.join([','.join([str(e) for e in v]) for v in side.vertices])
            svg += f'<polyline points="{pts}" style="fill:none; stroke:#{colors[i]}; stroke-width:1.5" />'
        pts = ' '.join([','.join([str(e) for e in v]) for v in self.vertices + [self.vertices[0]]])
        svg += f'<polyline points="{pts}" style="fill:#bbbbbb; stroke-width:0" />'
        # draw in a small circle for each corner (the first and last vertex for each side)
        for i, side in enumerate(self.sides):
            v = side.vertices[0]
            svg += f'<circle cx="{v[0]}" cy="{v[1]}" r="{1.0}" style="fill:#000000; stroke-width:0" />'
        svg += '</svg>'
        with open(full_svg_path, 'w') as f:
            f.write(svg)

        for i, side in enumerate(self.sides):
            side_path = os.path.join(output_path, f"side_{self.id}_{i}.yaml")
            data = {'piece_id': self.id, 'side_index': i, 'vertices': [list(v) for v in side.vertices], 'piece_center': list(side.piece_center), 'is_edge': side.is_edge}
            with open(side_path, 'w') as f:
                f.write(yaml.dump(data, default_flow_style=True))

    def find_border_raster(self) -> None:
        self.border = [[0 for i in range(self.width)] for j in range(self.height)]
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                v = self.pixels[y][x]
                if v == 0:
                    continue

                neighbors = [
                    self.pixels[y - 1][x],  # above
                    self.pixels[y + 1][x],  # below
                    self.pixels[y][x - 1],  # left
                    self.pixels[y][x + 1],  # right
                ]
                if not all(neighbors):
                    self.border[y][x] = 1

    def vectorize(self) -> None:
        """
        We want to "wind" a string around the border
        So we find the top-left most border pixel, then sweep a polyline around the border
        As we go, we capture how much the angle of the polyline changes at each step
        """
        sx, sy = None, None

        # start at the first border pixel we find
        for y, row in enumerate(self.border):
            if any(row):
                sy = y
                sx = row.index(1)
                break

        if sx is None or sy is None:
            raise Exception(f"Piece @ {self.id} has no border to walk")

        # print(f"Starting at {sx}, {sy}")
        self.vertices = [(sx, sy)]
        cx, cy = sx, sy
        p_angle = 0

        closed = False
        while not closed:
            neighbors = [
                # Clockwise sweep
                (cx,     cy - 1),  # above
                (cx + 1, cy - 1),  # above right
                (cx + 1, cy),      # right
                (cx + 1, cy + 1),  # below right
                (cx,     cy + 1),  # below
                (cx - 1, cy + 1),  # below left
                (cx - 1, cy),      # left
                (cx - 1, cy - 1),  # above left
            ]
            # pick where to start by looking at the current (absolute) angle, and looking
            # "back over our left shoulder" at the neighbor most in that direction, then sweep CW around the neighbors
            shift = int(round(p_angle * float(len(neighbors))/(2 * math.pi)))  # scale to wrap the full circle over the number of neighbors

            bx, by = cx, cy

            # check each neighbor in order to see if it is also a border
            # once we find one that is a border, add a point to it and continue
            for i in range(0 + shift, 8 + shift):
                nx, ny = neighbors[i % len(neighbors)]
                n = self.border[ny][nx]
                if n == 1:
                    dx, dy = nx - cx, ny - cy
                    abs_angle = math.atan2(dy, dx)
                    rel_angle = abs_angle - p_angle
                    p_angle = abs_angle

                    if rel_angle < 0:
                        rel_angle += 2 * math.pi

                    self.vertices.append((cx, cy))

                    cx, cy = nx, ny
                    if cx == sx and cy == sy:
                        closed = True

                    break

            if bx == cx and by == cy:
                raise Exception(f"Piece @ {self.id} will get us stuck in a loop because the border goes up to the edge of the bitmap. Take a new picture with the piece centered better or make sure the background is brighter white.")

    def merge_close_points(self, vs, threshold):
        i = -len(vs)
        while i < len(vs):
            i = i % len(vs)
            j = (i + 1) % len(vs)
            if util.distance(vs[i], vs[j]) <= threshold:
                min_v = util.midpoint_along_path(vs, vs[i], vs[j])

                # if the two points are next to each other, don't take the latter one, or else we'll then advance along the path indefinitely if there are more neighboring points
                if min_v == vs[j]:
                    min_v = vs[i]

                vs[i] = min_v
                vs.pop(j)
            else:
                i += 1

    def simplify(self) -> None:
        # first, let's remove any duplicate points
        vs = list(dict.fromkeys(self.vertices))

        # save the original vertices
        self.all_vertices = [(v[0], v[1]) for v in vs]

        # create a denser set of vertices that is higher res
        self.dense_vertices = [(v[0], v[1]) for v in vs]
        # self.dense_vertices = util.ramer_douglas_peucker(self.dense_vertices, epsilon=0.25)
        self.merge_close_points(self.dense_vertices, threshold=MERGE_IF_CLOSER_THAN_PX)

        # simplify and merge close points
        vs = util.ramer_douglas_peucker(vs, epsilon=SIMPLIFY_EPSILON)
        self.merge_close_points(vs, threshold=MERGE_IF_CLOSER_THAN_PX)

        self.vertices = vs
        self.centroid = util.centroid(self.vertices)

    def find_four_corners(self):
        # let's further simplify the shape, because this will lengthen the gap between vertices on flat parts
        vertices = util.ramer_douglas_peucker(self.all_vertices, epsilon=2 * SIMPLIFY_EPSILON)
        self.merge_close_points(vertices, threshold=3 * MERGE_IF_CLOSER_THAN_PX)
        self.corners = []

        # to find a corner, we're going to compute the angle between 3 consecutive points
        # if it is roughly 90º and pointed toward the center, it's a corner
        for i in range(len(vertices)):
            h = (i - 1) % len(vertices)
            j = (i + 1) % len(vertices)
            p_h = vertices[h]
            p_i = vertices[i]
            p_j = vertices[j]

            # sides will have relatively low curvature, so let's expect vertices to be spaced out if we're on a corner
            if util.distance(p_h, p_i) < 5 or util.distance(p_i, p_j) < 5:
                continue

            a_h = util.angle_between(p_i, p_h)
            a_j = util.angle_between(p_i, p_j)
            a_c = util.angle_between(p_i, self.centroid)
            d_hj = util.compare_angles(a_h, a_j)
            d_hc = util.compare_angles(a_h, a_c)
            d_jc = util.compare_angles(a_j, a_c)

            print(f"{i} \t {round(a_h * 180 / math.pi)} \t {round(a_j * 180 / math.pi)} \t {round(a_c * 180 / math.pi)}")
            print(f"\t {round(d_hj * 180 / math.pi)} \t {round(d_hc * 180 / math.pi)} \t {round(d_jc * 180 / math.pi)}")

            # See if the delta between the two legs is roughly 90º
            is_roughly_90 = abs(d_hj - math.pi/2) < SIDES_ORTHOGONAL_THRESHOLD_DEG * math.pi/180

            # See if it's pointed toward the center by saying the angle between vector to center and to a given leg is
            # smaller than the total angle, meaning the vector to center is between the vector ij and vector ih
            is_pointed_toward_center = d_hc < d_hj and d_jc < d_hj
            is_pointed_toward_center &= d_hc < 65 * math.pi/180 and d_jc < 65 * math.pi/180

            print(f"\t {is_roughly_90} \t {is_pointed_toward_center}")
            is_corner = is_roughly_90 and is_pointed_toward_center
            if is_corner:
                print(f"Found corner at {i}")
                self.corners.append(p_i)

        self.vertices = vertices  # set for debugging
        self.render()

        self.vertices = self.all_vertices
        self.render()

        if len(self.corners) != 4:
            self.vertices = simplified_vertices  # set for debugging
            raise Exception(f"Expected 4 corners, found {len(self.corners)} on piece {self.id}")

    def insert_corners(self) -> None:
        """
        The corners aren't necessarily existing vertices, so we need to insert them into the list of vertices
        """
        for corner in self.corners:
            # loop through our vertices until we find either an exact match (then we do nothing)
            # or we find the closest insertion point, and add it in there
            match = False
            min_d_ij = float("inf")
            min_i = None
            for i in range(len(self.vertices)):
                v_i = self.vertices[i]
                v_j = self.vertices[(i + 1) % len(self.vertices)]
                if corner == v_i:
                    match = True
                    break

                # we can't just find the closest vertex, because we might need to insert before that one
                # so we find the closest pair of vertices and insert between them
                d_i = util.distance(corner, v_i)
                d_j = util.distance(corner, v_j)
                d_sum = d_i + d_j
                if d_sum < min_d_ij:
                    min_d_ij = d_sum
                    min_i = i

            if match:
                # this corner is already in the list of vertices
                print(f"Corner {corner} already exists in vertices")
                continue

            # otherwise, add after index min_i
            self.vertices.insert(min_i + 1, corner)
            print(f"Inserted corner {corner} after index {min_i}")

    def find_four_sides(self) -> None:
        """
        Once we've found the corners, we'll identify which vertices belong to each side
        We do some validation to make sure the geometry of the piece is sensible
        """
        self.sides = []

        # first we have to figure out where our corners should be inserted as vertices

        # new_side = sides.Side(piece_id=self.id, side_id=None, vertices=self._pull_vertices_between(vs[0], vs[-1], self.dense_vertices), piece_center=self.center, is_edge=is_edge)
        # self.sides.append(new_side)

        # we need to find 4 sides
        if len(self.sides) != 4:
            raise Exception(f"Expected 4 sides, found {len(self.sides)} on piece {self.id}")

        # opposite sides should be parallel
        if abs(self.sides[0].angle - self.sides[2].angle) > SIDE_PARALLEL_THRESHOLD_DEG:
            raise Exception(f"Expected sides 0 and 2 to be parallel, but they are not ({self.sides[0].angle - self.sides[2].angle})")

        if abs(self.sides[1].angle - self.sides[3].angle) > SIDE_PARALLEL_THRESHOLD_DEG:
            raise Exception(f"Expected sides 1 and 3 to be parallel, but they are not ({self.sides[1].angle - self.sides[3].angle})")

        # make sure that sides 0 and 1 are roughly at a right angle
        if abs(self.sides[1].angle - self.sides[0].angle - 90 * math.pi/180.0) >  SIDES_ORTHOGONAL_THRESHOLD_DEG:
            raise Exception(f"Expected sides 0 and 1 to be at a right angle, but they are not ({self.sides[1].angle} - {self.sides[0].angle})")

        lengths = [s.length for s in self.sides]
        len_ratio02 = abs(lengths[0] - lengths[2])/lengths[0]
        len_ratio13 = abs(lengths[1] - lengths[3])/lengths[1]
        if len_ratio02 > 0.35 or len_ratio13 > 0.35:
            raise Exception(f"Expected sides to be roughly the same length, but they are not ({len_ratio02}, {len_ratio13})")

        d02 = util.distance_between_segments(self.sides[0].segment, self.sides[2].segment)
        d13 = util.distance_between_segments(self.sides[0].segment, self.sides[2].segment)
        if d02 > 1.35 * d13 or d13 > 1.35 * d02:
            raise Exception(f"Expected the piece to be roughly square, but the distance between sides is not comparable ({d02} vs {d13})")

        edge_count = sum([s.is_edge for s in self.sides])
        if edge_count > 2:
            raise Exception(f"A piece cannot be a part of more than 2 edges, found {edge_count}")
        elif edge_count == 2:
            if (self.sides[0].is_edge and self.sides[2].is_edge) or (self.sides[1].is_edge and self.sides[3].is_edge):
                raise Exception("A piece cannot be a part of two edges that are parallel!")

    def render(self) -> None:
        SIDE_COLORS = [util.RED, util.GREEN, util.PURPLE, util.CYAN]
        CORNER_COLOR = util.YELLOW
        lines = []
        for row in self.pixels:
            line = []
            for pixel in row:
                line.append('# ' if pixel == 1 else '. ')
            lines.append(line)

        for i, (px, py) in enumerate(self.vertices):
            color = util.BLACK_ON_WHITE  # orphans
            for si, side in enumerate(self.sides):
                if side.p1 == (px, py) or side.p2 == (px, py):
                    color = util.YELLOW
                    break
                elif (px, py) in side.vertices:
                    color = SIDE_COLORS[si % len(SIDE_COLORS)]
                    break
            if py >= len(lines) or px >= len(lines[py]):
                continue
            lines[py][px] = f"{color}{str(i % 10)} {util.WHITE}"

        print(f' {util.GRAY} ' + 'v ' * self.width + f"{util.WHITE}")
        for line in lines:
            line = ''.join(line)
            print(f'{util.GRAY}> {util.WHITE}{line}{util.GRAY}<{util.WHITE}')
        print(f' {util.GRAY} ' + '^ ' * self.width + f"{util.WHITE}")

    @property
    def center(self) -> Tuple[int, int]:
        minx = min([x for x,_ in self.vertices])
        maxx = max([x for x,_ in self.vertices])
        miny = min([y for _,y in self.vertices])
        maxy = max([y for _,y in self.vertices])
        return (minx + maxx) // 2, (miny + maxy) // 2

    def compare(self, piece) -> float:
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
                error = self.sides[p0_side_i].error_when_fit_with(piece[p1_side_i], flip=False, render=False)
                if error is None:
                    cumulative_error = None
                    break
                else:
                    cumulative_error += error

            if min_cumulative_error is None or (cumulative_error is not None and cumulative_error < min_cumulative_error):
                min_cumulative_error = cumulative_error

        return min_cumulative_error
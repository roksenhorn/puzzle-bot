import math
import os
import yaml
from PIL import Image
from typing import List, Tuple

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
        with Image.open(filename) as img:
            # Get image data as a 2D array of pixels
            width, height = img.size
            pixels = list(img.getdata())

        binary_pixels = []

        # Convert pixels to 0 or 1
        for i, pixel in enumerate(pixels):
            x = i % width
            y = i // width
            if y >= len(binary_pixels):
                binary_pixels.append([])
            binary_pixels[y].append(1 if pixel > 0 else 0)

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
            self.find_four_sides()
            self.extend_sides_to_corners()
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

    def should_join_to_snake(self, snake, point) -> bool:
        angle_tail = util.angle_between(snake[-2], snake[-1])
        angle_to_candidate = util.angle_between(snake[-1], point)
        return util.compare_angles(angle_tail, angle_to_candidate) < SIDE_PARALLEL_THRESHOLD_DEG * math.pi/180

    def _closest_vertex(self, p, vs):
        min_dist = None
        min_i = None
        for i, v in enumerate(vs):
            dist = util.distance(p, v)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_i = i
        return (min_dist, min_i)

    def _pull_vertices_between(self, p1, p2, vs):
        _, min_index1 = self._closest_vertex(p1, vs)
        _, min_index2 = self._closest_vertex(p2, vs)
        return util.slice(vs, min_index1, min_index2)

    def find_four_sides(self) -> None:
        """
        Finds the 4 vertices that represent the corners of the piece
        A side either has 2 vertices if it is a perfectly flat edge,
        or it has 4 vertices if there is a prong sticking out or poking in (2 adjacent, then a few vertices for the prong, then 2 more vertices)
        """
        self.sides = []

        # grow snakes as long as they don't have any sharp bends
        vs = list(self.vertices)
        vs.append(vs[0])
        snakes = []
        while len(vs) > 0:
            snake = [vs.pop(0), vs.pop(0)]
            continue_snake = True
            while continue_snake and len(vs) > 0:
                candidate = vs[0]
                if self.should_join_to_snake(snake, candidate):
                    snake.append(vs.pop(0))
                    continue_snake = True
                else:
                    # let the next snake start with the last point of this snake
                    vs.insert(0, snake[-1])
                    continue_snake = False
            snakes.append(snake)

        # see if we should connect the last snake to the first
        assert snakes[0][0] == snakes[-1][-1]
        if self.should_join_to_snake(snakes[-1], snakes[0][1]):
            merge = snakes[-1] + snakes[0][1:]
            snakes[0] = merge
            snakes.pop(-1)

        snakes = [s for s in snakes if util.distance(s[0], s[-1]) > 0.09 * self.dim]

        # for snake in snakes:
        #     print("Snake: ", snake, "len: ", util.distance(snake[0], snake[-1]))

        claimed_snakes = []

        # find snakes we're "collinear" with
        for snake in snakes:
            # if we're already dancing with someone else, we've been claimed and won't explore any further

            debug = False
            if snake[0] == (601, 2):
                print("\n>>>> ", snake)
                debug = True

            if snake in claimed_snakes:
                continue

            # see if each other snake is collinear with us
            for other_snake in snakes:
                if snake == other_snake:
                    continue

                # first, they can't be ridiculously far away
                gap = util.distance(snake[-1], other_snake[0])
                snakes_close_enough = (gap < 0.35 * self.dim)

                # they also need to sum up to enough of a flat edge
                total_len = util.distance(snake[0], snake[-1]) + util.distance(other_snake[0], other_snake[-1])
                enough_flat_side = total_len > 0.3 * self.dim

                # and last but not least, they need to be somewhat collinear
                # we define this as roughly the same angle, and their bounding boxes overlap
                angle_snake = util.angle_between(snake[0], snake[-1])
                angle_other = util.angle_between(other_snake[0], other_snake[-1])
                angles_close_enough = util.compare_angles(angle_snake, angle_other) < 2 * SIDE_PARALLEL_THRESHOLD_DEG * math.pi/180

                # determine how colinear the two snakes are
                # if they are far apart, then the line that connects them should be roughly parallel to both snakes
                # if they are really close, then they just need to be roughly parallel to each other
                angle_bridge = util.angle_between(snake[-1], other_snake[0])
                bridge_angle_between = util.compare_angles(angle_snake, angle_bridge) < SIDE_PARALLEL_THRESHOLD_DEG * math.pi/180
                bridge_angle_between = bridge_angle_between and util.compare_angles(angle_other, angle_bridge) < 1.3 * SIDE_PARALLEL_THRESHOLD_DEG * math.pi/180
                dist_between_snakes = util.distance(snake[-1], other_snake[0])
                really_close = dist_between_snakes < 8

                collinear = angles_close_enough and (bridge_angle_between or really_close)

                if debug:
                    print(other_snake)
                    print(f"gap: {gap} < {0.35 * self.dim}, total_len: {total_len} > {0.3 * self.dim}, angle_snake {angle_snake}, {angle_other}, {angle_bridge} angles_close_enough: {angles_close_enough}, bridge_angle_between: {bridge_angle_between}, really_close: {really_close}, collinear: {collinear}")

                # let's make sure we'd join these two snakes with the line that connects them
                if snakes_close_enough and enough_flat_side and collinear:
                    # print("We've found a partner!", snake, other_snake)
                    claimed_snakes.append(snake)
                    claimed_snakes.append(other_snake)

                    i = self.vertices.index(snake[0])
                    j = self.vertices.index(other_snake[-1])

                    # check if this is an edge or not
                    vs = util.slice(self.vertices, i, j)
                    dist_i_j = util.distance(vs[0], vs[-1])
                    side_perimeter = 0
                    for k in range(len(vs) - 1):
                        side_perimeter += util.distance(vs[k], vs[k+1])

                    # if the perimeter of the side is approx equal to the dist from start to end, it's a flat edge
                    if abs(side_perimeter - dist_i_j) < 0.04 * (side_perimeter + dist_i_j)/2.0:
                        is_edge = True
                    else:
                        is_edge = False

                    new_side = sides.Side(piece_id=self.id, side_id=None, vertices=self._pull_vertices_between(vs[0], vs[-1], self.dense_vertices), piece_center=self.center, is_edge=is_edge)
                    self.sides.append(new_side)
                    break

        # for any that weren't paired up, see if they are in fact a complete flat edge
        for snake in snakes:
            # ignore any that are a part of a non-edge side
            if snake in claimed_snakes:
                continue

            # we say a snake is long enough to be an edge if it is the majority of that dimension
            # so if it lies flat, it should be most of the width
            # if it is vertical, it should be most of the height
            snake_len = util.distance(snake[0], snake[-1])
            snake_angle = util.angle_between(snake[0], snake[-1])
            dim = self.height + (self.width - self.height) * (1 + math.cos(2 * snake_angle)) / 2

            if snake_len > EDGE_WIDTH_MIN_RATIO * dim:
                # print("Snake:", snake, "len:", round(snake_len), "@ angle:", round(snake_angle * 180/math.pi), "new dim:", round(dim), self.width, self.height)
                new_side = sides.Side(piece_id=self.id, side_id=None, vertices=self._pull_vertices_between(snake[0], snake[-1], self.dense_vertices), piece_center=self.center, is_edge=True)
                self.sides.append(new_side)

        # put in clockwise order
        sorted_sides = sorted(self.sides, key=lambda s: s.angle)
        self.sides = []

        # drop any sides that don't connect near the other sides
        # print("RESULT:")
        for i, s in enumerate(sorted_sides):
            k = (i - 1) % len(sorted_sides)
            j = (i + 1) % len(sorted_sides)
            prior_side = sorted_sides[k]
            next_side = sorted_sides[j]

            if util.distance(prior_side.vertices[-1], s.vertices[0]) <= 10 \
                or util.distance(next_side.vertices[0], s.vertices[-1]) <= 10:
                self.sides.append(s)
            #     print(" - ", s)
            # else:
            #     print(f"dropping {s}")

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
        if len_ratio02 > 0.3 or len_ratio13 > 0.3:
            raise Exception(f"Expected sides to be roughly the same length, but they are not ({len_ratio02}, {len_ratio13})")

        d02 = util.distance_between_segments(self.sides[0].segment, self.sides[2].segment)
        d13 = util.distance_between_segments(self.sides[0].segment, self.sides[2].segment)
        if d02 > 1.25 * d13 or d13 > 1.25 * d02:
            raise Exception(f"Expected the piece to be roughly square, but the distance between sides is not comparable ({d02} vs {d13})")

        edge_count = sum([s.is_edge for s in self.sides])
        if edge_count > 2:
            raise Exception(f"A piece cannot be a part of more than 2 edges, found {edge_count}")
        elif edge_count == 2:
            if (self.sides[0].is_edge and self.sides[2].is_edge) or (self.sides[1].is_edge and self.sides[3].is_edge):
                raise Exception("A piece cannot be a part of two edges that are parallel!")

    def extend_sides_to_corners(self) -> None:
        """
        Grow the sides to the corners of the piece
        """
        for corner in range(4):
            j = (corner - 1) % 4
            side_after = self.sides[corner]
            side_before = self.sides[j]

            intersection = util.intersection((side_before.vertices[-5], side_before.vertices[-2]), (side_after.vertices[1], side_after.vertices[4]))
            corner_vi = self.all_vertices.index(side_after.p1)

            closest_i = None
            closest_dist = None
            for i in range(corner_vi - 4, corner_vi + 5):
                v = self.all_vertices[i % len(self.all_vertices)]
                dist = util.distance(v, intersection)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    closest_i = i % len(self.all_vertices)

                if v in side_before.vertices:
                    j = side_before.vertices.index(v)
                    side_before.vertices.pop(j)
                if v in side_after.vertices:
                    j = side_after.vertices.index(v)
                    side_after.vertices.pop(j)

            # print(f"Corner {corner} is at {side_after.p1} (index {corner_vi}), intersection is {intersection}")
            # print(f"\tClosest point to intersection is {self.all_vertices[closest_i]} (@ {closest_i}), distance is {closest_dist}")
            corner_vertex = self.all_vertices[closest_i]
            side_before.vertices.append(corner_vertex)
            side_after.vertices.insert(0, corner_vertex)
            side_before.recompute_endpoints()
            side_after.recompute_endpoints()

        self.vertices = []
        for i, side in enumerate(self.sides):
            if i == 0:
                self.vertices.extend(side.vertices)
            else:
                self.vertices.extend(side.vertices[1:])
        self.vertices = self.vertices[:-1]

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
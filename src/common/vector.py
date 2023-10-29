import itertools
import json
import math
import os
from typing import List

from common import sides, util


# How much error to tolerate when simplifying the shape
SIMPLIFY_EPSILON = 1.5

# We'll merge vertices closer than this
MERGE_IF_CLOSER_THAN_PX = 1.75

# Opposing sides must be "parallel" within this threshold (in degrees)
SIDE_PARALLEL_THRESHOLD_DEG = 32

# Adjacent sides must be "orthogonal" within this threshold (in degrees)
SIDES_ORTHOGONAL_THRESHOLD_DEG = 42

# A side must be at least this long to be considered an edge
EDGE_WIDTH_MIN_RATIO = 0.4


class Candidate(object):
    def __init__(self, v, i, centroid, corner_angle=10000, offset_from_center=10000, stdev=10000):
        self.v = v
        self.i = i
        self.centroid = centroid
        self.angle = corner_angle
        self.stdev = stdev
        self.offset_from_center = offset_from_center

    def score(self):
        # lower score = better candidate for a corner
        # we determine "worst" by a mix of:
        # - how far from 90º the join angle is
        # - how far from the center the corner "points": the midangle of the corner typically points quite close to the center of the piece
        # - how non-straight the spokes are

        # how much bigger are we than 90º?
        # If we're less, then we're more likely to be a corner so we let this go below zero
        angle_error = self.angle - math.pi/2
        score = (1.0 * angle_error) + (0.5 * self.offset_from_center) + (0.7 * self.stdev)
        return score

    def __repr__(self) -> str:
        return f"Candidate(v={self.v}, i={self.i}, angle={round(self.angle * 180/math.pi, 1)}°, orientation offset={round(self.offset_from_center * 180/math.pi, 1)}°, stdev={self.stdev}, score={self.score()}"

    def __eq__(self, __value: object) -> bool:
        return (self.v[0] == __value.v[0]) and (self.v[1] == __value.v[1])

    def __hash__(self) -> int:
        return hash(self.v[0]) + hash(self.v[1])


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
        self.corners = []

    def process(self, output_path=None, render=False):
        print(f"> Vectorizing {self.id}")
        self.find_border_raster()
        self.vectorize()

        try:
            self.find_four_corners()
            self.extract_four_sides()
        except Exception as e:
            self.render()
            print(f"Error while processing {self.id}:")
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
            svg += f'<polyline points="{pts}" style="fill:none; stroke:#{colors[i]}; stroke-width:1.0" />'
        # draw in a small circle for each corner (the first and last vertex for each side)
        for i, side in enumerate(self.sides):
            v = side.vertices[0]
            svg += f'<circle cx="{v[0]}" cy="{v[1]}" r="{1.0}" style="fill:#000000; stroke-width:0" />'
        svg += f'<circle cx="{self.centroid[0]}" cy="{self.centroid[1]}" r="{1.0}" style="fill:#990000; stroke-width:0" />'
        svg += '</svg>'
        with open(full_svg_path, 'w') as f:
            f.write(svg)

        for i, side in enumerate(self.sides):
            side_path = os.path.join(output_path, f"side_{self.id}_{i}.json")
            data = {'piece_id': self.id, 'side_index': i, 'vertices': [list(v) for v in side.vertices], 'piece_center': list(side.piece_center), 'is_edge': side.is_edge}
            with open(side_path, 'w') as f:
                f.write(json.dumps(data))

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

        self.centroid = util.centroid(self.vertices)

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

    def find_four_corners(self):
        candidates = self.find_candidates_by_distance_maxima_from_center()
        candidates += self.find_corner_candidates_by_spoke_angles()
        candidates = self.nudge_to_angular_corners(candidates)
        self.select_best_corners(candidates)

    def find_candidates_by_distance_maxima_from_center(self):
        """
        Looks for local maximas in the distance from the centroid to the vertices
        """
        distances = [util.distance(v, self.centroid) for v in self.vertices]
        candidates = []
        for i in range(len(distances)):
            prev_distances = util.slice(distances, i - 7, i - 1)
            next_distances = util.slice(distances, i + 1, i + 7)
            if distances[i] >= max(prev_distances) and distances[i] >= max(next_distances):
                candidate = Candidate(v=self.vertices[i], i=i, centroid=self.centroid)
                candidates.append(candidate)

        return candidates

    def nudge_to_angular_corners(self, candidates):
        """
        Sometimes the local maxima is not at the corner of two spokes, but rather just along one of the spokes
        Let's nudge it to the best corner vertex and compute the angle that this corner represents
        """

        # eliminate duplicates
        candidates = list(set(candidates))

        def angle_at(i):
            """
            Compute the angle of the path at the given index by looking ±4 points away
            """
            h = (i - 4) % len(self.vertices)
            i = i % len(self.vertices)
            j = (i + 4) % len(self.vertices)
            v_h = self.vertices[h]
            v_i = self.vertices[i]
            v_j = self.vertices[j]
            angle = util.counterclockwise_angle_between_vectors(v_h, v_i, v_j)

            a_ic = util.angle_between(v_i, self.centroid)
            opens_toward_angle = util.angle_between(v_i, util.midpoint(v_h, v_j))
            offset_from_center = util.compare_angles(opens_toward_angle, a_ic)

            return angle, offset_from_center

        # Look a few before and a few after to see if a nearby vertex is a better corner candidate
        refined_candidates = []
        for candidate in candidates:
            debug = (candidate.v[1] < -5)
            if debug:
                print(candidate)

            subcandidates = []
            for j in range(-4, 5):
                angle, offset_from_center, stdev = self.compute_angle_and_stdev(self.vertices, candidate.i + j)
                i = (candidate.i + j) % len(self.vertices)
                subcandidate = Candidate(v=self.vertices[i], i=i, centroid=self.centroid, corner_angle=angle, offset_from_center=offset_from_center, stdev=stdev)
                if debug:
                    v = self.vertices[i]
                    print(f"\t{j}:    ({v[0]}, {v[1]}) \t {subcandidate}")
                subcandidates.append(subcandidate)

            # pick the best subcandidate
            best = min(subcandidates, key=lambda c: c.score())
            if debug:
                print("Best: " + str(best))
            refined_candidates.append(best)

        return refined_candidates

    def compute_angle_and_stdev(self, vertices, i):
        i = i % len(self.vertices)
        v_i = self.vertices[i]

        # find the angle from i to the points before it (h), and i to the points after (j)
        vec_offset = 1  # we start comparing to this many points away, as really short vectors have noisy angles
        vec_len = 12  # compare this many total points

        # see how straight the spokes are from this point, and what angle they jut out at
        a_ih, stdev_h = util.colinearity(from_point=vertices[i], to_points=util.slice(vertices, i-vec_len-vec_offset, i-vec_offset-1))
        a_ij, stdev_j = util.colinearity(from_point=vertices[i], to_points=util.slice(vertices, i+vec_offset+1, i+vec_len+vec_offset))

        # extend out along the avg spoke direction
        p_h = (v_i[0] + 10 * math.cos(a_ih), v_i[1] + 10 * math.sin(a_ih))
        p_j = (v_i[0] + 10 * math.cos(a_ij), v_i[1] + 10 * math.sin(a_ij))

        # how wide is the angle between the two legs?
        angle_hij = util.counterclockwise_angle_between_vectors(p_h, v_i, p_j)

        a_ic = util.angle_between(v_i, self.centroid)
        opens_toward_angle = util.angle_between(v_i, util.midpoint(p_h, p_j))
        offset_from_center = util.compare_angles(opens_toward_angle, a_ic)

        return angle_hij, offset_from_center, (stdev_h + stdev_j)/2

    def select_best_corners(self, candidates):
        """
        We look at the sharpness and the relative position to figure out which candidates make
        the best set of 4 corners

        We first compute a goodness score for each individual corner
        Then we find which set of 4 corners has the best cumulative score, where we'll weigh
        individual corner scores, plus how evenly spread out the 4 corners are (radially)
        """
        # eliminate duplicates
        candidates = list(set(candidates))

        # compute each corner's score, and only consider the n best corners
        candidates = sorted(candidates, key=lambda c: c.score())[:8]

        def _score_4_candidates(cs):
            """
            Given a set of 4 candidate corners, we produce a unitless score for how good they are
            Lower is better
            """
            # first, start with the score of each individual corner (how close ot 90º it is, etc)
            score = sum([c.score() for c in cs])

            # now, we want the corners to be roughly evenly spaced radially around the piece
            radial_positions = sorted([util.angle_between(self.centroid, c.v) for c in cs])
            max_radial_gap = 0
            min_radial_gap = 2 * math.pi
            side_lens = []
            for i in range(4):
                p1 = radial_positions[i]
                p2 = radial_positions[(i + 1) % 4]
                angle_between = util.compare_angles(p1, p2)
                if angle_between > max_radial_gap:
                    max_radial_gap = angle_between
                if angle_between < min_radial_gap:
                    min_radial_gap = angle_between

                side_len = util.distance(cs[i].v, cs[(i + 1) % 4].v)
                side_lens.append(side_len)

            # we want them to be roughly evenly spaced radially around the piece
            biggest_gap_penalty = max((max_radial_gap - math.pi/2), 0)
            smallest_gap_penalty = max((math.pi/2 - min_radial_gap), 0)
            score += biggest_gap_penalty + smallest_gap_penalty

            # we also want them to form a rectangle, where opposing sides are roughly the same length (not polyline length, but straight line length)
            len_ratio_s0_s2 = abs(1.0 - side_lens[0] / side_lens[2])
            len_ratio_s1_s3 = abs(1.0 - side_lens[1] / side_lens[3])
            len_ratio_penalty = 0.5 * (len_ratio_s0_s2 + len_ratio_s1_s3)
            score += len_ratio_penalty

            return cs, score

        # Generate all combinations of 4 corners and score them each up
        all_combinations = list(itertools.combinations(candidates, 4))
        all_scores = sorted([_score_4_candidates(c) for c in all_combinations], key=lambda c: c[1])
        best_corners = sorted(all_scores[0][0], key=lambda c: util.angle_between(self.centroid, c.v))
        self.corners = [c.v for c in best_corners]
        self.corner_indices = [c.i for c in best_corners]

        # the four corners should be roughly 90º from each other
        # we'll use the angle between the spokes to determine this
        corner_angles = [util.angle_between(self.centroid, c) for c in self.corners]
        for i in range(4):
            j = (i + 1) % 4
            angle = corner_angles[i]
            angle_next = corner_angles[j]
            angle_between = util.compare_angles(angle, angle_next)
            delta_90 = util.compare_angles(math.pi/2, angle_between)
            if delta_90 > 45 * math.pi/180:
                raise Exception(f"Corner angles are not roughly 90º: {round(angle * 180 / math.pi)}º and {round(angle_next * 180 / math.pi)}º = {round(angle_between * 180 / math.pi)}º apart, on piece {self.id}")

    def find_corner_candidates_by_spoke_angles(self):
        """
        Unused
        """
        vertices = self.vertices
        candidates = []

        # to find a corner, we're going to compute the angle between 3 consecutive points
        # if it is roughly 90º and pointed toward the center, it's a corner
        for i in range(len(vertices)):
            p_i = vertices[i]
            debug =(p_i[1] == 1047)
            if debug:
                print(f"\n\n\n!!!!!!!!!!!!!! {p_i} !!!!!!!!!!!!!!!\n\n\n")

            # find the angle from i to the points before it (h), and i to the points after (j)
            vec_offset = 1  # we start comparing to this many points away, as really short vectors have noisy angles
            vec_len = 12  # compare this many total points
            a_ih, stdev_h = util.colinearity(from_point=p_i, to_points=util.slice(vertices, i-vec_len-vec_offset, i-vec_offset-1), debug=debug)
            a_ij, stdev_j = util.colinearity(from_point=p_i, to_points=util.slice(vertices, i+vec_offset+1, i+vec_len+vec_offset), debug=debug)
            stdev = (stdev_h + stdev_j)/2

            if stdev > 0.2:
                continue

            if debug:
                print(f"a_ih: {round(a_ih * 180/math.pi)} stdev_h: {stdev_h}")
                print(f"a_ij: {round(a_ij * 180/math.pi)} stdev_j: {stdev_j}")
                print(util.slice(vertices, i-vec_len-vec_offset, i-vec_offset-1))
                print(util.slice(vertices, i+vec_offset+1, i+vec_len+vec_offset))

            p_h = (p_i[0] + 10 * math.cos(a_ih), p_i[1] + 10 * math.sin(a_ih))
            p_j = (p_i[0] + 10 * math.cos(a_ij), p_i[1] + 10 * math.sin(a_ij))

            # how wide is the angle between the two legs?
            angle_hij = util.counterclockwise_angle_between_vectors(p_h, p_i, p_j)

            # See if the delta between the two legs is roughly 90º
            is_roughly_90 = abs(math.pi/2 - angle_hij) < SIDES_ORTHOGONAL_THRESHOLD_DEG * math.pi/180
            if not is_roughly_90:
                if debug:
                    print(f"not roughly 90º: {round(angle_hij * 180/math.pi)}")
                continue

            # corners generally open up toward the center of the piece
            # meaning the mid-angle that comes out of the corner is roughly the same as the angle from the corner to the center
            angle_ih = util.angle_between(p_i, p_h)
            a_ic = util.angle_between(p_i, self.centroid)
            opens_toward_angle = util.angle_between(p_i, util.midpoint(p_h, p_j))
            offset_from_center = util.compare_angles(opens_toward_angle, a_ic)
            is_pointed_toward_center = offset_from_center < angle_hij / 2

            is_corner = is_roughly_90 and is_pointed_toward_center
            if is_corner:
                candidate = Candidate(v=p_i, i=i, centroid=self.centroid, corner_angle=angle_hij, offset_from_center=offset_from_center)
                candidates.append(candidate)
            elif debug:
                print(f"NOT a possible corner at {i}: {p_i} w/ {stdev}")
                print(f"\t c={self.centroid} \t ih={round(angle_ih * 180 / math.pi)} \t ic={round(a_ic * 180 / math.pi)} \t ij={round(a_ij * 180 / math.pi)} \t offset={round(offset_from_center * 180 / math.pi)}")
                print(f"\t angle width: {round(angle_hij * 180 / math.pi)}")
                print(f"\t roughly 90? {is_roughly_90} \t toward center? {is_pointed_toward_center} \t open toward {round(opens_toward_angle * 180 / math.pi)} \t offset from center {round(offset_from_center * 180 / math.pi)}\n")

        # we often find a handful of points right near the corner
        # let's go through and pick the best of those
        eliminated_corner_indices = []
        for c in candidates:
            if c.i in eliminated_corner_indices:
                # if we've been ruled out, no work to do
                continue

            for c2 in candidates:
                # if we're close to another candidate, figure out if we're the better corner
                if c.i != c2.i and util.distance(c.v, c2.v) < 10:
                    # choose the corner thats furthest from the center of the piece, meaning it juts out the most
                    d_centroid_i = util.distance(c.v, self.centroid)
                    d_centroid_j = util.distance(c2.v, self.centroid)
                    if d_centroid_i >= d_centroid_j:
                        eliminated_corner_indices.append(c2.i)

        candidates = [c for c in candidates if c.i not in eliminated_corner_indices]

        if len(candidates) < 4:
            print(eliminated_corner_indices)
            raise Exception(f"Expected 4 corners, but only found {len(candidates)} on piece {self.id}")

        return candidates

    def extract_four_sides(self) -> None:
        """
        Once we've found the corners, we'll identify which vertices belong to each side
        We do some validation to make sure the geometry of the piece is sensible
        """
        self.sides = []

        for i in range(4):
            # yank out all the relevant vertices for this side
            j = (i + 1) % 4
            c_i = self.corner_indices[i]
            c_j = self.corner_indices[j]
            vertices = util.slice(self.vertices, c_i, c_j)

            # do a bit of simplification
            vertices = util.ramer_douglas_peucker(vertices, epsilon=0.25)
            self.merge_close_points(vertices, threshold=MERGE_IF_CLOSER_THAN_PX)

            # make sure to include the true endpoints
            if vertices[0] != self.corners[i]:
                vertices.insert(0, self.corners[i])
            if vertices[-1] != self.corners[j]:
                vertices.append(self.corners[j])

            # edges are flat and thus have very little variance from the average line between all points
            corner_corner_distance = util.distance(vertices[0], vertices[-1])
            polyline_length = util.polyline_length(vertices)
            is_edge = polyline_length / corner_corner_distance < 1.1

            side = sides.Side(piece_id=self.id, side_id=None, vertices=vertices, piece_center=self.centroid, is_edge=is_edge)
            self.sides.append(side)

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
        avg02 = (lengths[0] + lengths[2])/2
        avg13 = (lengths[1] + lengths[3])/2
        len_ratio02 = abs(lengths[0] - lengths[2])/avg02
        len_ratio13 = abs(lengths[1] - lengths[3])/avg13
        if len_ratio02 > 0.55 or len_ratio13 > 0.55:
            print(f"Lengths: {lengths}; ratios: {len_ratio02}, {len_ratio13}")
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

        i = 0
        for si, side in enumerate(self.sides):
            for (px, py) in side.vertices:
                color = SIDE_COLORS[si % len(SIDE_COLORS)]
                if py >= len(lines) or px >= len(lines[py]):
                    continue
                lines[py][px] = f"{color}{str(i % 10)} {util.WHITE}"

        for (px, py) in self.corners:
            value = lines[py][px].split(' ')[0][-1]
            lines[py][px] = f"{util.BLACK_ON_BLUE}{value} {util.WHITE}"

        lines[self.centroid[1]][self.centroid[0]] = f"{util.BLACK_ON_RED}X {util.WHITE}"

        print(f' {util.GRAY} ' + 'v ' * self.width + f"{util.WHITE}")
        for i, line in enumerate(lines):
            line = ''.join(line)
            print(f'{util.GRAY}> {util.WHITE}{line}{util.GRAY}<{util.WHITE} {i}')
        print(f' {util.GRAY} ' + '^ ' * self.width + f"{util.WHITE}")

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
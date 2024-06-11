import os
import argparse
import random
import math
from shapely.geometry import LineString, Point
import numpy as np


class Side:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.bezier = [start, start, end, end]
        self.max_height = 0

    @staticmethod
    def random_side(side_start, side_end, max_height):
        if random.random() < 0.15:
            return Side.sinusoid(side_start, side_end, max_height)
        else:
            return Side.side_with_nub(side_start, side_end, random.choice([-1, 1]), max_height)

    @staticmethod
    def sinusoid(side_start, side_end, max_height):
        side = Side(side_start, side_end)

        nubby_sinusoid = random.random() < 0.5
        if nubby_sinusoid:
            num_half_periods = random.randint(2, 5)
        else:
            num_half_periods = random.randint(1, 5)
        amplitude = random.uniform(0.1, 0.2) / (num_half_periods**0.5)
        phase = random.choice([-1, 1])
        angle = math.atan2(side_end[1] - side_start[1], side_end[0] - side_start[0])

        bezier = [(0, 0), (0, 0)]
        for i in range(num_half_periods):
            if nubby_sinusoid:
                hump_width = random.uniform(0.75, 1.5) / num_half_periods
            else:
                hump_width = random.uniform(0.2, 0.4) / num_half_periods
            pointx = (i + 1) / (num_half_periods + 1)
            pointy = amplitude * (1 if i % 2 == 0 else -1) * phase
            bezier.append((pointx - hump_width, pointy))
            bezier.append((pointx, pointy))
            bezier.append((pointx + hump_width, pointy))
            print(i, pointx, pointy)

        bezier += [(1, 0), (1, 0)]

        # scale the bezier to the right length
        length = math.sqrt((side_end[0] - side_start[0])**2 + (side_end[1] - side_start[1])**2)
        bezier = [(x * length, y * length) for (x, y) in bezier]

        # # rotate the bezier curve to the correct angle
        bezier = [(x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)) for (x, y) in bezier]

        # translate the bezier curve to the correct start point
        bezier = [(side_start[0] + x, side_start[1] + y) for (x, y) in bezier]

        side.bezier = bezier
        side.max_height = max_height
        return side

    @staticmethod
    def side_with_nub(side_start, side_end, nub_direction, max_height):
        side = Side(side_start, side_end)

        # construct a unit nub and annotate how each point can be manipulated
        points = [
            (0.000, 1.000), # anchor
            (0.238, 1.000), # control point1
            (0.357, 0.900), # control point2
            (0.357, 0.760), # anchor
            (0.357, 0.590), # control point1
            (0.048, 0.590), # control point2
            (0.048, 0.305), # anchor
            (0.048, 0.076), # ...
            (0.250, 0.000),
            (0.500, 0.000),
            (0.750, 0.000),
            (1.000, 0.124),
            (1.000, 0.286),
            (1.000, 0.476),
            (0.695, 0.508),
            (0.695, 0.760),
            (0.695, 0.943),
            (0.881, 1.000),
            (1.000, 1.000),
        ]
        points = [(x, y - 1) for (x, y) in points]

        # randomize the width of the neck
        # negative shifts the left side of the neck to the left and right side to the right, making it wider
        # positive shifts the left side of the neck to the right and right side to the left, making it narrower
        neck_width_shift = random.uniform(-0.25, 0.03)
        for i in range(0, 5):
            points[i] = (points[i][0] + neck_width_shift, points[i][1])
            points[-i - 1] = (points[-i - 1][0] - neck_width_shift, points[-i - 1][1])

        mid_width_shift = random.uniform(-0.25, 0.2) + 0.5 * neck_width_shift
        for i in range(5, 9):
            points[i] = (points[i][0] + mid_width_shift, points[i][1])
            points[-i - 1] = (points[-i - 1][0] - mid_width_shift, points[-i - 1][1])

        # add a random skew to the left or right for some nubs
        skew = random.uniform(-0.5, 0.5) if random.random() < 0.25 else 0.0
        for i, (x, y) in enumerate(points):
            points[i] = (x + skew * y, y)

        # randomize the height of the nub
        height_mult = random.uniform(0.9, 1.4 if neck_width_shift < 0.0 else 1.6)
        points = [(x, y * height_mult) for (x, y) in points]

        # sometimes make the nub pointier or clefted at the top
        if random.random() < 0.15:
            if mid_width_shift > 0.0:
                # for narrow nubs, make them pointy
                pointy = random.uniform(0.05, 0.25)
                for i in range(8, 11):
                    points[i] = (points[i][0], points[i][1] - pointy)
            else:
                # for wide nubs, make them clefted
                cleft = random.uniform(-0.35, -0.1)
                points[8] = (points[8][0] + 0.09, points[8][1] - cleft)
                points[9] = (points[9][0], points[9][1] - cleft)
                points[10] = (points[10][0] - 0.09, points[10][1] - cleft)

        # sometimes have the nub go out, sometime have the nub go in
        if nub_direction == -1:
            points = [(x, -y) for (x, y) in points]

        scale = 0.25 * max_height
        scaled = [(scale * x, scale * y) for (x, y) in points]

        # rotate around the origin by a random amount centerd around the slope of the side
        theta = math.atan((side_end[1] - side_start[1]) / (side_end[0] - side_start[0])) + random.uniform(-0.2, 0.2)
        if theta < - math.pi / 4:
            theta += math.pi
        rotated = [(x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)) for (x, y) in scaled]

        # 0.0 is all the way to the left, 1.0 is all the way to the right
        left_right_ratio = random.uniform(0.15, 0.85)
        is_vertical = theta > math.pi / 4 and theta < 3 * math.pi / 4
        nub_width = max([x for (x, _) in rotated]) - min([x for (x, _) in rotated]) if not is_vertical else 0
        nub_height = max([y for (_, y) in rotated]) - min([y for (_, y) in rotated]) if is_vertical else 0
        translate_x = side_start[0] + left_right_ratio * (side_end[0] - side_start[0] - nub_width)
        translate_y = side_start[1] + left_right_ratio * (side_end[1] - side_start[1] - nub_height)
        # add some extra wiggling
        if is_vertical:  # vertical sides, nudge left or right
            translate_x += random.uniform(-150, 150) / max_height
        else:  # horizontal sides, nudge up or down
            translate_y += random.uniform(-150, 150) / max_height

        translated = [(x + translate_x, y + translate_y) for (x, y) in rotated]
        side.bezier = [side_start, side_start, translated[0]] + translated + [translated[-1], side_end, side_end]
        side.max_height = max_height
        return side

    @property
    def endpoints(self):
        return [
            (self.x, self.y),
            (self.x, self.y),
        ]

    @property
    def svg(self):
        path_data = f"M {self.bezier[0][0]} {self.bezier[0][1]}"  # Move to the first anchor point
        for i in range(1, len(self.bezier) - 2, 3):
            p1, p2, p3 = self.bezier[i], self.bezier[i+1], self.bezier[i+2]
            path_data += f" C {p1[0]} {p1[1]}, {p2[0]} {p2[1]}, {p3[0]} {p3[1]}"  # Cubic Bézier to the next anchor
        return f""" <path d="{path_data}" stroke="black" fill="none"/>"""

    def neighbors(self, sides):
        """Returns any sides that share an endpoint with this side"""
        neighbors = []
        for other in sides:
            if other == self:
                continue
            if self.start == other.start or self.start == other.end or self.end == other.start or self.end == other.end:
                neighbors.append(other)
        return neighbors

    def collides_with(self, other, r):
        """Check if two Bezier curves intersect within a given radius r."""

        # borders won't have collisions
        if len(self.bezier) == 4 or len(other.bezier) == 4:
            return False

        def bezier_point(b, t):
            """Compute a point on a Bezier curve defined by control points b at parameter t."""
            n = len(b) - 1
            p = np.array(b, dtype=float)
            for r in range(1, n + 1):
                p[:n - r + 1] = (1 - t) * p[:n - r + 1] + t * p[1:n - r + 2]
            return p[0]

        def bezier_curve_to_linestring(b, num_points=100):
            """Convert a Bezier curve to a LineString by sampling points."""
            points = [bezier_point(b, t) for t in np.linspace(0, 1, num_points)]
            return LineString(points)

        # Convert input lists to numpy arrays
        b1 = [np.array(point) for point in self.bezier[2:-3]]
        b2 = [np.array(point) for point in other.bezier[2:-3]]

        # Convert Bezier curves to LineStrings
        line1 = bezier_curve_to_linestring(b1)
        line2 = bezier_curve_to_linestring(b2)

        # Buffer the LineStrings by radius r
        buffer1 = line1.buffer(r)
        buffer2 = line2.buffer(r)

        # Check if the buffers intersect
        return buffer1.intersects(buffer2)



def generate(cols, rows, width, height, waviness, output_dir):
    gridpoints, sides = _generate(cols, rows, width, height, waviness)

    svg = f'<svg width="{width}" height="{height}" viewBox="-10 -10 {width} {height}" xmlns="http://www.w3.org/2000/svg">'

    for i, (x, y) in enumerate(gridpoints):
        point = gridpoints[(x, y)]
        hue = 360 * i / len(gridpoints)
        svg += f'<circle cx="{point[0]}" cy="{point[1]}" r="2" fill="hsl({hue}, 100%, 50%)" />'

    for (x, y), sides in sides.items():
        for side in sides:
            svg += side.svg

    svg += "</svg>"
    with open(os.path.join(output_dir, "puzzle.svg"), "w") as f:
        f.write(svg)

    print(f"Generated puzzle.svg in {output_dir}")


def _generate(cols, rows, width, height, waviness):
    """
    Lays out a grid for where each pieces' corners will be at
    """
    # scale waviness to prevent zeros
    W = 0.85
    MIN_FEATURE_PX = 0.08 * (width / cols)
    print(f"Min feature size: {MIN_FEATURE_PX} px")

    rows += 1
    cols += 1

    def random_floats(n, min, max, sum_to):
        """
        Generate n random floats between min and max that sum to sum_to
        """
        # generate n random numbers
        nums = [random.uniform(min, max) for _ in range(n)]
        # scale them so they sum to sum_to
        scale = sum_to / sum(nums)
        return [num * scale for num in nums]

    min_row_height = (1.0 - W * waviness**1.1) * float(height) / float(rows)
    max_row_height = (1.0 + W * waviness**1.1) * float(height) / float(rows)
    row_heights = random_floats(n=rows, min=min_row_height, max=max_row_height, sum_to=height)

    min_col_width = (1.0 - W * waviness**1.1) * float(width) / float(cols)
    max_col_width = (1.0 + W * waviness**1.1) * float(width) / float(cols)
    col_widths = random_floats(n=cols, min=min_col_width, max=max_col_width, sum_to=width)

    gridpoints = {}
    sides = {}
    for i in range(cols):
        for j in range(rows):
            if i == 0 or i == cols - 1 or j == 0 or j == rows - 1:
                x_wave = 0
                y_wave = 0
            else:
                x_wave = random.uniform(-1.0, 1.0) * W * waviness**0.9 * (width / cols)
                y_wave = random.uniform(-1.0, 1.0) * W * waviness**0.9 * (height / rows)
            x = sum(col_widths[0:i]) + x_wave
            y = sum(row_heights[0:j]) + y_wave
            gridpoints[(i, j)] = (x, y)

            sides[(i, j)] = []

            # generate horizontal sides
            if i > 0:
                prev_x, prev_y = gridpoints[(i - 1, j)]
                if j == 0 or j == rows - 1:
                    sides[(i, j)].append(Side((prev_x, prev_y), (x, y)))
                else:
                    max_height = min(row_heights[j], row_heights[j + 1])
                    sides[(i, j)].append(Side.random_side((prev_x, prev_y), (x, y), max_height))

            # generate vertical sides
            if j > 0:
                prev_x, prev_y = gridpoints[(i, j - 1)]
                if i == 0 or i == cols - 1:
                    sides[(i, j)].append(Side((prev_x, prev_y), (x, y)))
                else:
                    max_height = min(col_widths[i], col_widths[i + 1])
                    sides[(i, j)].append(Side.random_side((prev_x, prev_y), (x, y), max_height))

    # some sides will naturally intersect nearby sides
    # if that's the case, we regnerate the side until it doesn't conflict
    while False:
        conflict = False
        for (i, j), sides_list in sides.items():
            for side in sides_list:
                all_sides = [side for sides_list in sides.values() for side in sides_list]
                neighbors = side.neighbors(all_sides)
                for neighbor in neighbors:
                    while side.collides_with(neighbor, r=MIN_FEATURE_PX):
                        conflict = True
                        sides[(i, j)].remove(side)
                        side = Side.random_side(side.start, side.end, side.max_height)
                        print(f"Regenerating side at ({i}, {j})")
                        sides[(i, j)].append(side)
        if not conflict:
            break

    return gridpoints, sides


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a puzzle")
    parser.add_argument("--cols", type=int, default=10, help="How many columns wide")
    parser.add_argument("--rows", type=int, default=10, help="How many rows tall")
    parser.add_argument("--width", type=int, default=400, help="How many px wide")
    parser.add_argument("--height", type=int, default=400, help="How many px tall")
    parser.add_argument("--random-seed", type=int, default=1, help="How to seed random number generator")
    parser.add_argument("--waviness", type=float, default=0.3, help="[0.0, 1.0] How much to make the puzzle pieces wavy")
    parser.add_argument("--output-dir", type=str, default=".", help="Which dir to save the puzzle.svg")
    args = parser.parse_args()

    # consistent random results
    random.seed(args.random_seed)
    generate(args.cols, args.rows, args.width, args.height, args.waviness, args.output_dir)
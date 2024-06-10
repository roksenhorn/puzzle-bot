import os
import argparse
import random
import math


class Nub:
    def __init__(self, side_start, side_end, nub_direction, max_height):
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
            (0.190, 0.000),
            (0.5, 0.000),
            (0.890, 0.000),
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
        skew = random.uniform(-0.4, 0.4) if random.random() < 0.25 else 0.0
        for i, (x, y) in enumerate(points):
            points[i] = (x + skew * y, y)

        # randomize the height of the nub
        height_mult = random.uniform(0.9, 1.4 if neck_width_shift < 0.0 else 1.6)
        points = [(x, y * height_mult) for (x, y) in points]

        # sometimes have the nub go out, sometime have the nub go in
        if nub_direction == -1:
            points = [(x, -y) for (x, y) in points]

        scale = 0.25 * max_height
        scaled = [(scale * x, scale * y) for (x, y) in points]

        # rotate around the origin by a random amount centerd around the slope of the side
        theta = math.atan((side_end[1] - side_start[1]) / (side_end[0] - side_start[0])) #+ random.uniform(-0.2, 0.2)
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
        translated = [(x + translate_x, y + translate_y) for (x, y) in rotated]
        self.bezier = [side_start, side_start, translated[0]] + translated + [translated[-1], side_end, side_end]

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


def generate(cols, rows, width, height, waviness, output_dir):
    gridpoints, sides = _generate_grid(cols, rows, width, height, waviness)

    svg = f'<svg width="{width}" height="{height}" viewBox="-10 -10 {width} {height}" xmlns="http://www.w3.org/2000/svg">'

    for i, (x, y) in enumerate(gridpoints):
        point = gridpoints[(x, y)]
        hue = 360 * i / len(gridpoints)
        svg += f'<circle cx="{point[0]}" cy="{point[1]}" r="2" fill="hsl({hue}, 100%, 50%)" />'

    for side in sides:
        svg += side

    svg += "</svg>"
    with open(os.path.join(output_dir, "puzzle.svg"), "w") as f:
        f.write(svg)

    print(f"Generated puzzle.svg in {output_dir}")


def _generate_grid(cols, rows, width, height, waviness):
    """
    Lays out a grid for where each pieces' corners will be at
    """
    # scale waviness to prevent zeros
    W = 0.85

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
    print(row_heights)
    print(col_widths)

    gridpoints = {}
    sides = []
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

            # generate horizontal sides
            if i > 0:
                prev_x, prev_y = gridpoints[(i - 1, j)]
                if j == 0 or j == rows - 1:
                    border_side_svg = f""" <line x1="{prev_x}" y1="{prev_y}" x2="{x}" y2="{y}" stroke="black"/>"""
                    sides.append(border_side_svg)
                else:
                    nub_direction = random.choice([-1, 1])
                    max_height = row_heights[j] if nub_direction == -1 else row_heights[j + 1]
                    nubbded_side = Nub((prev_x, prev_y), (x, y), nub_direction, max_height).svg
                    sides.append(nubbded_side)

            # generate vertical sides
            if j > 0:
                prev_x, prev_y = gridpoints[(i, j - 1)]
                if i == 0 or i == cols - 1:
                    border_side_svg = f""" <line x1="{prev_x}" y1="{prev_y}" x2="{x}" y2="{y}" stroke="black"/>"""
                    sides.append(border_side_svg)
                else:
                    nub_direction = random.choice([-1, 1])
                    max_height = col_widths[i] if nub_direction == -1 else col_widths[i + 1]
                    nubbded_side = Nub((prev_x, prev_y), (x, y), nub_direction, max_height).svg
                    sides.append(nubbded_side)

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
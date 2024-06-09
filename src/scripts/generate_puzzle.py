import os
import argparse
import random


def generate(cols, rows, width, height, waviness, output_dir):
    gridpoints, sides = _generate_grid(cols, rows, width, height, waviness)

    svg = f'<svg width="{width}" height="{height}" viewBox="-10 -10 {width} {height}" xmlns="http://www.w3.org/2000/svg">'

    for i, (x, y) in enumerate(gridpoints):
        point = gridpoints[(x, y)]
        hue = 360 * i / len(gridpoints)
        svg += f'<circle cx="{point[0]}" cy="{point[1]}" r="2" fill="hsl({hue}, 100%, 50%)" />'

    for i, nub in enumerate(sides):
        hue = 360 * i / len(sides)
        linestr = " ".join([f"{x},{y}" for x, y in nub])
        svg += f'<polyline points="{linestr}" fill="none" stroke="hsl({hue}, 100%, 50%)" stroke-width="1" />'

    svg += "</svg>"
    with open(os.path.join(output_dir, "puzzle.svg"), "w") as f:
        f.write(svg)

    print(f"Generated puzzle.svg in {output_dir}")


# 0.5 => 0.2
# 0.2 => 0.5

def _generate_horizontal_nubbed_side(ax, ay, bx, by, waviness, max_y):
    """
    Generate a side with a nub that connects the two points
    """
    w = bx - ax
    slope = (by - ay) / w

    nub_base_width = 0.15 + random.uniform(0.0, 0.4)
    nub_base_center_x = (0.5 + random.uniform(-0.4, 0.4) * waviness)
    nub_start_x = ax + nub_base_center_x * (bx - ax) - nub_base_width/2 * (bx - ax)
    nub_end_x = ax + nub_base_center_x * (bx - ax) + nub_base_width/2 * (bx - ax)
    nub_slope = slope * random.uniform(0.5, 1.5) * waviness
    nub_start_y = ay + nub_slope * (nub_start_x - ax)
    nub_end_y = ay + nub_slope * (nub_end_x - ax)

    nub_direction = random.choice([-1, 1])
    maxy = max(nub_start_y, nub_end_y)
    miny = min(nub_start_y, nub_end_y)
    nub_ratio_factor = 1.0 - nub_base_width  # if the nub is wide, it should be short, if the nub is narrow, it should be tall
    nub_peak_y = (maxy if nub_direction == 1 else miny) + nub_direction * (0.3 * max_y * nub_ratio_factor * random.uniform(0.6, 1.4))
    nub_peak_x = ax + nub_base_center_x * (bx - ax)
    return [
        (ax, ay),
        (nub_start_x, nub_start_y),
        (nub_peak_x, nub_peak_y),
        (nub_end_x, nub_end_y),
        (bx, by),
    ]


def _generate_vertical_nubbed_side(ax, ay, bx, by, waviness):
    return [
        (ax, ay),
        (bx, by),
    ]


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

    min_row_height = (1.0 - W * waviness) * float(height) / float(rows)
    max_row_height = (1.0 + W * waviness) * float(height) / float(rows)
    row_heights = random_floats(n=rows, min=min_row_height, max=max_row_height, sum_to=height)

    min_col_width = (1.0 - W * waviness) * float(width) / float(cols)
    max_col_width = (1.0 + W * waviness) * float(width) / float(cols)
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
                x_wave = random.uniform(-1, 1) * W * waviness * (width / cols)
                y_wave = random.uniform(-1, 1) * W * waviness * (height / rows)
            x = sum(col_widths[0:i]) + x_wave
            y = sum(row_heights[0:j]) + y_wave
            gridpoints[(i, j)] = (x, y)

            # generate horizontal sides
            if i > 0:
                prev_x, prev_y = gridpoints[(i - 1, j)]
                if j == 0 or j == rows - 1:
                    border_side = [(prev_x, prev_y), (x, y)]
                    sides.append(border_side)
                else:
                    nubbded_side = _generate_horizontal_nubbed_side(prev_x, prev_y, x, y, waviness, max_y=row_heights[j])
                    sides.append(nubbded_side)

            # generate vertical sides
            if j > 0:
                prev_x, prev_y = gridpoints[(i, j - 1)]
                if i == 0 or i == cols - 1:
                    border_side = [(prev_x, prev_y), (x, y)]
                    sides.append(border_side)
                else:
                    nubbded_side = _generate_vertical_nubbed_side(prev_x, prev_y, x, y, waviness)
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
import math
from PIL import Image
from typing import List, Tuple
from shapely.geometry import Polygon, Point, LineString
import numpy as np


YELLOW = '\033[33m'
BLUE = '\033[34m'
GRAY = '\033[90m'
RED = '\033[31m'
GREEN = '\033[32m'
CYAN = '\033[36m'
PURPLE = '\033[35m'
WHITE = '\033[0m'
BLACK_ON_WHITE = '\033[30;47m'
BLACK_ON_BLUE = '\033[30;44m'
BLACK_ON_RED = '\033[30;41m'
BLACK_ON_GREEN = '\033[30;42m'


def load_bmp_as_binary_pixels(path):
    """
    Given a bitmap image path, returns a 2D array of 1s and 0s
    """
    with Image.open(path) as img:
        width, height = img.size
        pixels = np.array(img.getdata())

        # if the image is RGB or RGBA, convert to binary
        if type(pixels[0]) != np.int64 and len(pixels[0]) >= 3:
            pixels = np.array([sum(p[:3]) / 3 for p in pixels])

    # Reshape to 2D and convert to 1s and 0s
    pixels = pixels.reshape((height, width))
    binary_pixels = np.where(pixels > 0, 1, 0).astype(np.int8)
    return binary_pixels, width, height


def binary_pixel_data_for_photo(path, threshold, max_width=None, crop_by=0):
    """
    Given a bitmap image path, returns a 2D array of 1s and 0s
    """
    with Image.open(path) as img:
        if max_width is not None and img.size[0] > max_width:
            try:
                img.thumbnail((max_width, img.size[1]), resample=Image.NEAREST)
            except Exception as e:
                print(f"Error resizing {path}")
                raise e
            if crop_by:
                img = img.crop((crop_by, crop_by, img.size[0] - crop_by, img.size[1] - crop_by))

        # Get image data as a 1D array of pixels
        width, height = img.size
        pixels = list(img.getdata())

    # Convert pixels to 0 or 1 2D array
    # turns piece pixels into 1s, background into 0s
    # turn saturated parts to background color
    # e.g. we are using hot pink duck tape to delineate the border of the table
    # we also see lens blur comes across as a saturated blue
    bg_color = 0
    piece_color = 1
    binary_pixels = np.empty(height, dtype=object)
    for i, rgb in enumerate(pixels):
        brightness = sum(rgb) // 3
        binary_pixel = bg_color if brightness <= threshold else piece_color
        x = i % width
        y = i // width
        if x == 0:
            row = np.empty(width, dtype=np.int8)
            binary_pixels[y] = row

        binary_pixels[y][x] = binary_pixel

    return binary_pixels, width, height


def remove_small_islands(pixels, min_size, ignore_islands_along_border=False, island_value=1):
    """
    Find and remove all islands of pixels that are smaller than some set number of pixels
    """
    # find all the islands
    lines = [[e for e in l] for l in pixels]
    islands = find_islands(lines, ignore_islands_along_border=ignore_islands_along_border, island_value=island_value)

    # sort islands (which is a list of list) by len of each island
    islands.sort(key=lambda i: len(i), reverse=True)

    # remove all other islands
    removed_count = 0
    for island in islands:
        if len(island) < min_size:
            removed_count += 1
            for x, y in island:
                pixels[y][x] = 0 if island_value == 1 else 1


def ramer_douglas_peucker(points, epsilon):
    """
    Simplifies a polyline using the Ramer-Douglas-Peucker algorithm.
    :param points: List of (x,y) tuples representing the polyline.
    :param epsilon: Maximum distance between the original polyline and the simplified polyline.
    :return: List of (x,y) tuples representing the simplified polyline.
    """
    dmax = 0
    index = 0
    for i in range(1, len(points) - 1):
        d = distance_to_line(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        rec_results1 = ramer_douglas_peucker(points[:index+1], epsilon)
        rec_results2 = ramer_douglas_peucker(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[-1]]


def distance_to_line(point, start, end):
    """
    Calculates the distance from a point to a line segment.
    :param point: Tuple (x,y) representing the point.
    :param start: Tuple (x,y) representing the start of the line segment.
    :param end: Tuple (x,y) representing the end of the line segment.
    :return: The distance from the point to the line segment.
    """
    if start == end:
        return distance(point, start)
    else:
        n = abs((end[1] - start[1]) * point[0] - (end[0] - start[0]) * point[1] + end[0] * start[1] - end[1] * start[0])
        d = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
        return n / d


def distance(point1, point2):
    """
    Calculates the distance between two points.
    :param point1: Tuple (x,y) representing the first point.
    :param point2: Tuple (x,y) representing the second point.
    :return: The distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance_between_segments(line1, line2):
    """
    Calculates the distance between two line segments.
    :param line1: Tuple ((x1,y1),(x2,y2)) representing the first line segment.
    :param line2: Tuple ((x3,y3),(x4,y4)) representing the second line segment.
    :return: The distance between the two line segments.
    """
    return min(distance_to_line(line1[0], line2[0], line2[1]),
               distance_to_line(line1[1], line2[0], line2[1]),
               distance_to_line(line2[0], line1[0], line1[1]),
               distance_to_line(line2[1], line1[0], line1[1]))


def angle_between(p1, p2):
    """
    Calculates the angle between two points.
    :param p1: Tuple (x,y) representing the first point.
    :param p2: Tuple (x,y) representing the second point.
    :return: The angle between the two points.
    """
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def compare_angles(angle1, angle2):
    # Convert angles to range [0, 2*pi)
    angle1 = angle1 % (2*math.pi)
    angle2 = angle2 % (2*math.pi)
    # Compute absolute difference between angles
    diff = abs(angle1 - angle2)
    # Return difference in range [0, pi)
    return min(diff, 2*math.pi - diff)


def centroid(polygon):
    """
    Calculates the centroid of a polygon.
    :param polygon: List of (x,y) tuples representing the polygon.
    :return: The centroid of the polygon.
    """
    poly = Polygon(polygon)
    centroid = poly.centroid
    return (int(round(centroid.x)), int(round(centroid.y)))


def incenter(polygon):
    """
    Finds the point inside the polygon furthest from the edges.
    This should be the best area to grip the piece by.
    """
    polygon = Polygon(polygon)

    c = polygon.centroid
    search_radius = 0.25 * (polygon.bounds[2] - polygon.bounds[0])
    stride = 8
    minx, miny, maxx, maxy = c.x - search_radius, c.y - search_radius, c.x + search_radius, c.y + search_radius
    points = np.array([Point(x, y) for x in range(int(minx), int(maxx), stride) for y in range(int(miny), int(maxy), stride)])

    max_distance = 0
    incenter = None
    for point in points:
        distance = polygon.exterior.distance(point)
        if distance > max_distance:
            max_distance = distance
            incenter = point
    return (int(round(incenter.x)), int(round(incenter.y)))


def intersection(line1, line2):
    """
    Calculates the intersection between two line segments.
    :param line1: Tuple ((x1,y1),(x2,y2)) representing the first line segment.
    :param line2: Tuple ((x3,y3),(x4,y4)) representing the second line segment.
    :return: The intersection between the two line segments.
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:
        return None
    else:
        xi = ((x3 - x4) * (x1 * y2 - y1 * x2) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        yi = ((y3 - y4) * (x1 * y2 - y1 * x2) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return int(round(xi)), int(round(yi))


def slice(l: List, i: int, j: int) -> List:
    """
    Grabs a slice from the list, from index i through j
    j can be lower than i, in which case the slice will "wrap" around the list
    """
    if i < 0:
        i += len(l)
    if j < 0:
        j += len(l)
    while i >= len(l):
        i -= len(l)
    while j >= len(l):
        j -= len(l)
    if i < j:
        return l[i:j + 1]
    else:
        return l[i:] + l[:j + 1]


def counterclockwise_angle_between_vectors(h, i, j):
    """
    Calculates the angle between two vectors, i->h and i->j, sweeping counter-clockwise
            j
    h
           θ <--- angle
           i
    """
    # shift the vectors to be based at i
    v1 = (h[0] - i[0], h[1] - i[1])
    v2 = (j[0] - i[0], j[1] - i[1])

    # normalize the lengths
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(v1, v2)
    determinant = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.arctan2(determinant, dot_product)

    # Ensure the angle is in the range [0, 2π]
    if angle < 0:
        angle += 2 * np.pi

    # make CCW
    angle = 2 * np.pi - angle

    return float(angle)



def rotate(v: Tuple[int, int], around: Tuple[int, int], angle: float) -> Tuple[int, int]:
    """
    Rotates a vertex around a point by a given angle
    """
    x, y = v
    x0, y0 = around
    return (
        int(round((x - x0) * math.cos(angle) - (y - y0) * math.sin(angle) + x0)),
        int(round((x - x0) * math.sin(angle) + (y - y0) * math.cos(angle) + y0))
    )


def mirrored(vertices) -> List[Tuple[int, int]]:
    """
    Returns a list of vertices that have been geometrically mirrored around the x-axis
    """
    mirrored = []
    for v in vertices:
        mirrored.append((v[0], -v[1]))
    return mirrored


def resample_polyline(polyline, n):
    """
    Given a polyline and a number of points to resample to,
    returns a resampled polyline with n segments, each of equal length
    """
    line = LineString(polyline)
    line_length = line.length

    # Create n evenly spaced points along the line
    # and a list of Points at those distances
    distances = np.linspace(0, line_length, n + 1)
    points = [line.interpolate(distance) for distance in distances]
    return [(float(point.x), float(point.y)) for point in points], line_length


def error_between_polylines(polyline1, polyline2, p1_len):
    """
    Returns the total integrated error between two polylines
    """
    def _error_between_polylines(p1, p2):
        differences = np.abs(p1 - p2)
        error = np.sum(differences)
        error_x, error_y = np.sum(differences - p1 + p2, axis=0) / len(p1)
        return error, error_x, error_y

    # sample along the polylines at fixed intervals
    error, error_x, error_y = _error_between_polylines(polyline1, polyline2)

    # only allow a little bit of y shifting, up to +/- 5 pixels
    error_y = max(-5, min(5, error_y))

    # we often have slight alignment errors because of differences in corner shape
    # find the mean error, and shift by that amount, then recompute
    polyline1_shifted = [(x - error_x, y - error_y) for (x, y) in polyline1]
    error_shifted, _, _ = _error_between_polylines(polyline1_shifted, polyline2)
    return min(error, error_shifted) / p1_len, (error_x, error_y)


def distance_to_polyline(point, polyline):
    """
    Returns the distance from a point to a polyline, and that closest point on the polyline
    """
    min_dist = None
    closest_point = None
    for i in range(len(polyline) - 1):
        dist, p = _distance_to_segment(point, polyline[i], polyline[i + 1])
        if min_dist is None or dist < min_dist:
            min_dist = dist
            closest_point = p
    return min_dist, closest_point


def _distance_to_segment(point, p1, p2):
    """
    Computes the distance from a point to a line segment
    Returns the distance and the closest point
    """
    x, y = point
    x1, y1 = p1
    x2, y2 = p2

    # if the line segment is a point, return the distance to that point
    if x1 == x2 and y1 == y2:
        return distance(point, p1), p1

    # otherwise, compute the distance to the line
    u = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x0 = x1 + u * (x2 - x1)
    y0 = y1 + u * (y2 - y1)

    return distance(point, (x0, y0)), (x0, y0)


def point_at_dist_along_segment(p1, p2, dist):
    """
    Returns the point that is dist away from p1 along the line segment p1-p2
    """
    length = distance(p1, p2)
    if length == 0:
        return p1

    ratio = dist / length
    x = p1[0] + (p2[0] - p1[0]) * ratio
    y = p1[1] + (p2[1] - p1[1]) * ratio
    return (x, y)


def polyline_length(polyline):
    """
    Returns the length of a polyline
    """
    length = 0
    for i in range(len(polyline) - 1):
        length += distance(polyline[i], polyline[i + 1])
    return length


def polygonize(vertices) -> List[Tuple[int, int]]:
    """
    Close off the polyline by forming a quadrilateral from the first and last vertices that contains all other vertices
    Assumes the input vertices are rotated, starting at the origin, and positive x-values
    """
    stroked = LineString(vertices).buffer(1.0).exterior.coords
    return stroked


def scale(vertices, scale) -> List[Tuple[float, float]]:
    """
    Scales the vertices along the x axis to be w pixels wide
    """
    return [(float(x) * scale, y) for (x, y) in vertices]


def bounds(vertices) -> Tuple[int, int, int, int]:
    """
    Returns minx, miny, maxx, maxy
    """
    xs = [x for x,_ in vertices]
    ys = [y for _,y in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def tight_bounds(poly1, poly2):
    """
    Returns the tightest bounding box that contains both polygons
    """
    bounds1 = bounds(poly1)
    bounds2 = bounds(poly2)
    return (
        max(bounds1[0], bounds2[0]),
        max(bounds1[1], bounds2[1]),
        min(bounds1[2], bounds2[2]),
        min(bounds1[3], bounds2[3]),
    )


def midpoint(p1, p2):
    """
    Returns the midpoint between two points
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def midpoint_along_path(vertices, p1, p2) -> Tuple[int, int]:
    """
    Given a path of vertices, and 2 points that are a part of that path, find an existing "midpoint" vertex between them
    Finds the "midpoint" between two points that is a part of our border
    """
    min_max = None
    min_v = None
    for zz, v in enumerate(vertices):
        d1 = distance(v, p1)
        d2 = distance(v, p2)
        if min_max is None or max(d1, d2) <= min_max:
            min_max = max(d1, d2)
            min_v = v
    return min_v


def average_of_angles(angles):
    angles = np.array(angles)
    avg_x = np.mean(np.cos(angles))
    avg_y = np.mean(np.sin(angles))

    avg_angle = np.arctan2(avg_y, avg_x)
    return float(avg_angle)


def angular_stdev(angles):
    angles = np.array(angles)
    unit_vectors_x = np.cos(angles)
    unit_vectors_y = np.sin(angles)

    mean_x = np.mean(unit_vectors_x)
    mean_y = np.mean(unit_vectors_y)
    mean_angle = np.arctan2(mean_y, mean_x)

    angular_deviations = np.arctan2(np.sin(angles - mean_angle), np.cos(angles - mean_angle))
    stdev = np.std(angular_deviations)

    return float(stdev)


def curve_score(points, debug=False) -> float:
    """
    `point` is the point we're checking
    Returns a score of how likely this point is on a curve that opens toward the centroid, rather than a corner
    e.g. the center of this:
    after ->    /
    point ->   *          ==> curve
    befor ->    \

         |
         *     => not a curve
        /

    Returns 0 if it's a corner or even ending outwards, and 1 if it's on a perfect circle
    """
    if len(points) < 16:
        raise Exception("Need a bunch of points to calculate curve score")

    gap = 4
    angles = [counterclockwise_angle_between_vectors(points[i - gap], points[i], points[i + gap]) for i in range(gap, len(points) - gap)]
    avg_angle = average_of_angles(angles)
    angle_deviations = [compare_angles(angle, avg_angle) for angle in angles]

    # sum up the change in angle, weighing them more the closer to the center of the list of points they are
    # i=0 has 0 weight, i=len-1 has 0 weight, i=len/2 has weight 1.0
    weighted_deviation = 0.0
    mid_i = len(angles) // 2
    total_weight = 0.0
    for i, deviation in enumerate(angle_deviations):
        weight = 1 - abs(i - mid_i) / mid_i
        weighted_deviation += abs(deviation) * weight
        total_weight += weight

    normalized_deviation = weighted_deviation / total_weight
    score = 3 * (0.4 - normalized_deviation)
    score = max(min(1.0, score), 0.0)

    if debug:
        print(f"Num points: {len(points)}")
        print(points)
        print(f"Angle:")
        for a in angles:
            print(f"{round(a * 180/math.pi)}°", end="\t")
        print(f"\nAngle deviations:")
        for d in angle_deviations:
            print(f"{round(d * 180/math.pi)}°", end="\t")
        print(f"\nDeviation score: {weighted_deviation} / {total_weight} = {normalized_deviation}")
        print(f"Curve score: {score}")

    # Aggregate score
    return score


def colinearity(from_point, to_points, debug=False) -> Tuple[float, float]:
    """
    Given a point and a list of points, finds how "colinear" they are:
     - the average angle between the point and the points
     - the std dev of the angles
    """
    angles = [angle_between(from_point, to_point) for to_point in to_points]
    avg = average_of_angles(angles)
    std_dev = angular_stdev(angles)
    if debug:
        print("Colinearity:")
        print(f"angles: {[round(a * 180/math.pi) for a in angles]}")
        print(f"avg: {round(avg * 180/math.pi)}")
        print(f"deltas: {[(angle - avg) ** 2 for angle in angles]} \t ==> stdev: {std_dev}")
    return avg, std_dev


def sublist_exists(lst, sub_lst):
    """
    Returns true if sub_lst is a sublist of lst
    Allows for wrapping around the end of the list
    """
    if len(sub_lst) > len(lst):
        return False

    lst_extended = lst * 2
    sub_lst_str = ''.join(map(str, sub_lst))
    lst_extended_str = ''.join(map(str, lst_extended))

    return sub_lst_str in lst_extended_str

def find_islands(grid, callback=None, ignore_islands_along_border=False, island_value=1):
    """
    Given a grid of 0s and 1s, finds all "islands" of 1s:
    00000000
    01110000
    01111000
    00111110
    00000000

    :param grid: a 2D array of 0s and 1s
    :param callback: a function that will be called with each island found
    :param ignore_islands_along_border: if True, islands that touch the border of the grid will be ignored
    :param island_value: the value that represents an island in the grid (1 or 0)

    Returns either a list of islands, or a list of Trues if a callback was provided
    """
    visited1 = set()
    visited2 = set()
    islands = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1 and (i, j) not in visited1 and (i, j) not in visited2:
                island = set()
                queue = [(i, j)]
                touched_border = False

                # to prevent memory from getting too big and lookups from taking too long,
                # we maintain two visited sets, we check if we've visited a
                # location by checking either, and drain them offset
                if len(islands) % 160 == 0:
                    visited1 = set()
                if len(islands) % 160 == 80:
                    visited2 = set()
                while queue:
                    x, y = queue.pop(0)
                    if (x, y) not in visited1 and (x, y) not in visited2:
                        visited1.add((x, y))
                        visited2.add((x, y))
                        island.add((y, x))
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) and grid[x + dx][y + dy] == island_value:
                                queue.append((x + dx, y + dy))
                        if x == 0 or y == 0 or x == len(grid) - 1 or y == len(grid[0]) - 1:
                            touched_border = True

                if ignore_islands_along_border and touched_border:
                    continue

                if callback:
                    ok = callback(island, len(islands))
                    if ok:
                        islands.append(True)
                else:
                    islands.append(island)
    return islands


def render_polygons(vertices_list: List[List[Tuple[int, int]]], bounds=None) -> None:
    vertices_list =[[(int(round(x)), int(round(y))) for x, y in vs] for vs in vertices_list]

    # find the minx across all the vertices
    minx = min([min([x for x,_ in vertices]) for vertices in vertices_list])
    miny = min([min([y for _,y in vertices]) for vertices in vertices_list])
    maxx = max([max([x for x,_ in vertices]) for vertices in vertices_list])
    maxy = max([max([y for _,y in vertices]) for vertices in vertices_list])

    colors = [YELLOW, GREEN, BLUE, PURPLE, CYAN, RED]

    grid = []
    for i in range(miny, maxy + 1):
        row = []
        for j in range(minx, maxx + 1):
            row.append(' ')
        grid.append(row)

    polygons = [Polygon(vs) for vs in vertices_list]

    b_minx, b_miny, b_maxx, b_maxy = bounds if bounds else (minx, miny, maxx, maxy)
    minx = max(minx, b_minx)
    miny = max(miny, b_miny)
    maxx = min(maxx, b_maxx)
    maxy = min(maxy, b_maxy)

    # fill in the pixels that lay inside a polygon
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            for i, polygon in enumerate(polygons):
                if is_inside((x, y), polygon):
                    current_value = grid[y - miny][x - minx]
                    if current_value == ' ':
                        color = colors[i % len(colors)]
                    else:
                        color = GRAY
                    grid[y - miny][x - minx] = color + '.' + WHITE

    print(GRAY + '   ' + ' v' * (maxx - minx + 1) + WHITE + '\n')
    for row in grid:
        s = ' '.join(row)
        print(f'{GRAY}>   {WHITE}' + s + f'{GRAY}   <{WHITE}')
    print('\n   ' + GRAY + ' ^' * (maxx - minx + 1) + WHITE + '\n')


def is_inside(coord: Tuple[int, int], polygon: Polygon):
    point = Point(coord)
    return point.within(polygon) or point.touches(polygon)


def render_polylines(vertices_list: List[List[Tuple[int, int]]], bounds=None) -> None:
    vertices_list =[[(int(round(x)), int(round(y))) for x, y in vs] for vs in vertices_list]

    # find the minx across all the vertices
    minx = min([min([x for x,_ in vertices]) for vertices in vertices_list])
    miny = min([min([y for _,y in vertices]) for vertices in vertices_list])
    maxx = max([max([x for x,_ in vertices]) for vertices in vertices_list])
    maxy = max([max([y for _,y in vertices]) for vertices in vertices_list])

    colors = [YELLOW, GREEN, BLUE, PURPLE, CYAN, RED]

    grid = []
    for i in range(miny, maxy + 1):
        row = []
        for j in range(minx, maxx + 1):
            row.append(' ')
        grid.append(row)

    b_minx, b_miny, b_maxx, b_maxy = bounds if bounds else (minx, miny, maxx, maxy)
    minx = max(minx, b_minx)
    miny = max(miny, b_miny)
    maxx = min(maxx, b_maxx)
    maxy = min(maxy, b_maxy)

    for i, vertices in enumerate(vertices_list):
        for j, v in enumerate(vertices):
            color = colors[i % len(colors)]
            current_value = grid[v[1] - miny][v[0] - minx]
            if '.' in current_value or current_value == ' ':
                grid[v[1] - miny][v[0] - minx] = f'{color}{j % 10}{WHITE}'
            else:
                grid[v[1] - miny][v[0] - minx] = f'{BLACK_ON_WHITE}#{WHITE}'

    print(GRAY + '   ' + ' v' * (maxx - minx + 1) + WHITE + '\n')
    for i, row in enumerate(grid):
        s = ' '.join(row)
        print(f'{GRAY}>   {WHITE}' + s + f'{GRAY}   < {i}{WHITE}')
    print('\n   ' + GRAY + ' ^' * (maxx - minx + 1) + WHITE + '\n')


def normalized_ssd(bytes1, bytes2):
    """
    Computes the Sum of Squared Differences between two bytes
    Normalizes the result by the number of pixels

    Note: bytes1 and bytes2 must be the same length
    """
    assert len(bytes1) == len(bytes2)
    ssd = np.sum((bytes1 - bytes2) ** 2)
    return ssd / len(bytes1)


def normalized_area_between_corners(vertices):
    """
    Computes the area enclosed by a list of vertices,
    considering the area as positive when the points are non-collinear
    """
    sum_distances = 0
    p0 = vertices[0]
    p1 = vertices[-1]
    for p in vertices:
        sum_distances += distance_to_line(point=p, start=p0, end=p1)
    return sum_distances / len(vertices)


def remove_stragglers(pixels, width, height) -> bool:
    """
    Given an array of binary pixels, removes any pixels that are only connected to one other pixel
    Returns True if any were removed
    """
    removed = False

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            v = pixels[y][x]
            if v != 1:
                continue

            # All 8 neighbors
            above_left = pixels[y - 1][x - 1]
            above = pixels[y - 1][x]
            above_right = pixels[y - 1][x + 1]
            right = pixels[y][x + 1]
            below_right = pixels[y + 1][x + 1]
            below = pixels[y + 1][x]
            below_left = pixels[y + 1][x - 1]
            left = pixels[y][x - 1]
            neighbors = [
                above_left,
                above,
                above_right,
                right,
                below_right,
                below,
                below_left,
                left,
            ]
            borders = [True for n in neighbors if n == 1]
            if len(borders) <= 1:
                # straggler only connected by one
                pixels[y][x] = 0
                removed = True

            # if there are only 2 neighbors, and they are not adjacent (e.g. no [1, 1] subset in the list), then these
            # are one-pixel-wide bridges that should be removed
            if len(borders) == 2 and not sublist_exists(borders, [1, 1]):
                pixels[y][x] = 0
                removed = True

    if removed:
        return remove_stragglers(pixels, width, height)

    return False

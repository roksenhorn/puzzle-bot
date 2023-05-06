import math
from PIL import Image
from typing import List, Tuple
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.spatial import KDTree


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


def load_binary_image(path):
    """
    Given a bitmap image path, returns a 2D array of 1s and 0s
    """
    with Image.open(path) as img:
        # Get image data as a 1D array of pixels
        width, height = img.size
        pixels = list(img.getdata())

    # Convert pixels to 0 or 1 2D array
    binary_pixels = []
    for i, pixel in enumerate(pixels):
        x = i % width
        y = i // width
        if y >= len(binary_pixels):
            binary_pixels.append([])
        binary_pixels[y].append(1 if pixel > 0 else 0)

    return binary_pixels, width, height


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
    while i > len(l):
        i -= len(l)
    while j > len(l):
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


def chunk_polyline(polyline, num_chunks, t_width):
    """
    Given a polyline, a resulting number of sub-polyline chunks, and a width of the chunks,
    returns a dict mapping t to a polyline
    """
    chunk_length = max(4, math.ceil(len(polyline) * t_width))
    result = {}
    for tt in range(num_chunks):
        t = tt / (num_chunks - 1)
        mid_index = round(t * len(polyline))
        start_index = max(0, math.floor(mid_index - chunk_length / 2))
        end_index = min(len(polyline), math.ceil(mid_index + chunk_length / 2))
        if start_index == 0:
            end_index = chunk_length
        elif end_index == len(polyline):
            start_index = len(polyline) - chunk_length
        result[t] = polyline[start_index:end_index]
    return result


def total_integrated_error(polyline1, polyline2, find_zero_mean_error=True):
    """
    Returns the total integrated error between two polylines
    """
    p1_len = polyline_length(polyline1)

    NUM_STOPS = 26
    error = 0
    errors = []
    at_dist = 0
    i = 0

    p2_chunks = chunk_polyline(polyline2, NUM_STOPS, t_width=0.1)

    for tt in range(NUM_STOPS):
        t = tt / (NUM_STOPS - 1)

        # find the point on polyline1 at t
        p1_at_t, i, at_dist = _point_at_t_along_polyline(polyline1, p1_len, t, i, at_dist)

        # then find the sub-polyline of polyline2 around this t area
        relevant_p2_segments = p2_chunks[t]
        if len(relevant_p2_segments) < 2:
            print(f"ERROR: no relevant segments for t={t}, p2 segments={len(polyline2)}")
            for t, segments in p2_chunks.items():
                print(f"t={t}: {segments}")

        # then find the closest point on that sub-polyline to p1_at_t
        # we do this because we need to tolerate slight differences in lengths of the two polylines otherwise small imperfections early on acrue error along the whole polyline
        e_t, p = _distance_to_polyline(p1_at_t, relevant_p2_segments)
        error += e_t**1.5  # we want to weight the error more heavily when it's larger
        p_error = (p1_at_t[0] - p[0], p1_at_t[1] - p[1])  # x and y components of the error
        errors.append(p_error)

    if find_zero_mean_error:
        # we often have alignment errors
        # find the mean error, and shift and recompute by that amount
        X_SHIFT_MULT = 3.5  # much of the time, we'll have zero x error because the side is horizontal, but about 1/3 of the side will be vertical, so weight that portion higher
        mean_error = (sum([X_SHIFT_MULT * e[0] for e in errors]) / len(errors), sum([e[1] for e in errors]) / len(errors))

        # clamp the components of the mean error to be within [-2, 2]
        # if we allow too much shifting, we align the nub of the piece even if the corners don't line up at all
        mean_error = (min(2, max(-2, mean_error[0])), min(2, max(-2, mean_error[1])))

        polyline1_shifted = [(x - mean_error[0], y - mean_error[1]) for (x, y) in polyline1]
        error2, _ = total_integrated_error(polyline1_shifted, polyline2, find_zero_mean_error=False)
        error1 = error / p1_len
        return min(error1, error2), mean_error
    else:
        return error / p1_len, (0, 0)

    # TODO:
    # if we find some pieces just aren't matching great still, we can try rotating ± 2 degrees and see if that helps


def _distance_to_polyline(point, polyline):
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


def _point_at_t_along_polyline(polyline, length, t, i, at_dist):
    """
    Given a polyline and a t from 0 to 1, finds the point on the polyline at that t
    """
    target_dist = t * length

    while at_dist <= target_dist:
        next_dist = distance(polyline[i], polyline[i + 1])
        if at_dist + next_dist < target_dist:
            at_dist += next_dist
            i += 1
        else:
            # find the point along this next segment that gives us the remaining distance
            remaining_dist = target_dist - at_dist
            return point_at_dist_along_segment(polyline[i], polyline[i + 1], remaining_dist), i, at_dist


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


def find_islands(grid):
    visited = set()
    islands = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1 and (i, j) not in visited:
                island = set()
                queue = [(i, j)]
                while queue:
                    x, y = queue.pop(0)
                    if (x, y) not in visited:
                        visited.add((x, y))
                        island.add((y, x))
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) and grid[x + dx][y + dy] == 1:
                                queue.append((x + dx, y + dy))
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
        vertices = vertices[:-2]
        for j, v in enumerate(vertices):
            color = colors[i % len(colors)]
            current_value = grid[v[1] - miny][v[0] - minx]
            if '.' in current_value or current_value == ' ':
                grid[v[1] - miny][v[0] - minx] = f'{color}{j % 10}{WHITE}'
            else:
                grid[v[1] - miny][v[0] - minx] = f'{BLACK_ON_WHITE}#{WHITE}'

    print(GRAY + '   ' + ' v' * (maxx - minx + 1) + WHITE + '\n')
    for row in grid:
        s = ' '.join(row)
        print(f'{GRAY}>   {WHITE}' + s + f'{GRAY}   <{WHITE}')
    print('\n   ' + GRAY + ' ^' * (maxx - minx + 1) + WHITE + '\n')

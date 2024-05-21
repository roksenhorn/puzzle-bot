import math
from PIL import Image, ExifTags
from typing import List, Tuple
from shapely.geometry import Polygon, Point, LineString
import numpy as np
import numpy as np
from collections import deque
from scipy.ndimage import label, find_objects
from scipy import ndimage

from common.config import *


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

EXPECTED_PHOTO_ORIENTATION = 1 # Horizontal (normal)


def load_bmp_as_binary_pixels(path):
    """
    Given a bitmap image path, returns a 2D array of 1s and 0s
    """
    with Image.open(path) as img:
        width, height = img.size
        pixels = np.array(img.getdata())

        # if the image is RGB or RGBA, convert to binary
        if type(pixels[0]) not in [np.int64, np.int32] and len(pixels[0]) >= 3:
            pixels = np.array([sum(p[:3]) / 3 for p in pixels])

    # Reshape to 2D and convert to 1s and 0s
    pixels = pixels.reshape((height, width))
    binary_pixels = np.where(pixels > 0, 1, 0).astype(np.int8)
    return binary_pixels, width, height


def get_photo_orientation(img):
    exif = img._getexif()
    if exif:
        for tag, value in exif.items():
            if tag in ExifTags.TAGS:
                if ExifTags.TAGS[tag] == 'Orientation':
                    return value
    return None


def binary_pixel_data_for_photo(path, threshold, max_width=None, crop=None):
    """
    Given a bitmap image path, returns a 2D array of 1s and 0s
    """
    with Image.open(path) as img:
        if (orientation := get_photo_orientation(img)) is not None and orientation != EXPECTED_PHOTO_ORIENTATION:
            raise Exception(f"Image {path} is not oriented correctly: {orientation}")

        w, h = img.size
        if w < h:
            raise Exception(f"Image {path} is portrait, not landscape")

        if max_width is not None and img.size[0] > max_width:
            scale_factor = max_width / img.size[0]
            try:
                img = img.resize((max_width, int(img.size[1] * scale_factor)), resample=Image.NEAREST)
            except Exception as e:
                print(f"Error resizing {path}")
                raise e
        else:
            scale_factor = None

        if crop:
            w, h = img.size
            img = img.crop((crop[3], crop[0], w - crop[1], h - crop[2]))

        data, out_w, out_h = threshold_pixels(img, threshold)
        return data, out_w, out_h, scale_factor


def threshold_pixels(img, threshold):
    # Convert image to grayscale numpy array
    grayscale = img.convert('L')
    data = np.array(grayscale)

    # Apply threshold to get binary representation
    binary_data = np.where(data <= threshold, 0, 1).astype(np.int8)
    return binary_data, binary_data.shape[1], binary_data.shape[0]


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


def average_angles(angles):
    """
    Calculate the average angle from a list of angles, taking into account the circular nature of angles.
    We compute the average by converting each angle into its vector representation on the unit circle
    (using sin and cos), then averaging these vectors to find the resultant vector. The angle of this resultant
    vector is then converted back to degrees as the average angle. This method ensures that the average angle
    properly accounts for the wrap-around at 0 degrees (360 degrees).
    >>> average_angles([358, 2]) => 0.0
    >>> average_angles([90, 270]) => 180.0
    >>> average_angles([10, 20, 30]) => 20.0
    """
    if not angles:
        return None

    sum_sin = 0
    sum_cos = 0

    for angle in angles:
        angle = angle % 2 * math.pi
        # Sum up the sine and cosine values
        sum_sin += math.sin(angle)
        sum_cos += math.cos(angle)

    # Calculate average sine and cosine values
    avg_sin = sum_sin / len(angles)
    avg_cos = sum_cos / len(angles)

    # Calculate the average angle from the average sine and cosine
    if avg_cos == 0:
        if avg_sin > 0:
            average_angle = math.pi / 2  # 90 degrees
        else:
            average_angle = 3 * math.pi / 2  # 270 degrees
    else:
        average_angle = math.atan2(avg_sin, avg_cos)

    # Convert the average angle from radians back to degrees
    average_angle = math.degrees(average_angle)

    # Ensure the average angle is non-negative
    return average_angle if average_angle >= 0 else average_angle + 360


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


def rotate_polyline(polyline, around_point, angle):
    """
    Rotates a polyline around a point by a given angle.
    :param polyline: List of (x,y) tuples representing the polyline.
    :param around_point: Tuple (x,y) representing the point to rotate around.
    :param angle: The angle to rotate the polyline by.
    :return: List of (x,y) tuples representing the rotated polyline.
    """
    return [rotate(point, around_point, angle) for point in polyline]


def translate_polyline(polyline, translation):
    """
    Translates a polyline by a given translation.
    :param polyline: List of (x,y) tuples representing the polyline.
    :param translation: Tuple (x,y) representing the translation.
    :return: List of (x,y) tuples representing the translated polyline.
    """
    return [(point[0] + translation[0], point[1] + translation[1]) for point in polyline]


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
    # No two points may be the same. If they are, it's likely from noisy data and we should error.
    if (h == i) or (h == j) or (i == j):
        raise Exception(f"counterclockwise_angle_between_vectors: no points may be identical {h} {i} {j}")

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


def subtract(p1, p2):
    """
    Returns p1 - p2
    """
    return (p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2):
    """
    Returns the midpoint between two points
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def multimidpoint(ps):
    """
    Returns the midpoint between multiple points
    """
    x = sum([p[0] for p in ps]) / len(ps)
    y = sum([p[1] for p in ps]) / len(ps)
    return x, y


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


def find_islands(grid, ignore_islands_along_border=True, min_island_area=MIN_PIECE_AREA):
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

    Returns either a list of tuples: each island paired with its origin
    """
    # 8-connectivity - touching any other 1 on a side or corner
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

    # to find connected components
    labeled_array, num_features = label(grid, structure=structure)

    # Optional: Extract slices for each island
    slices = find_objects(labeled_array)
    islands = []
    origins = []
    for i, s in enumerate(slices, start=1):
        # Create a mask for the current island within the slice
        mask = (labeled_array[s] == i)
        # Apply the mask to the slice to extract only the current island
        island = grid[s] * mask
        islands.append(island)
        origins.append((s[1].start, s[0].start))

    # filter out any islands that touch the border (they're likely to be cropped pieces)
    if ignore_islands_along_border:
        h, w = grid.shape
        for i in range(num_features):
            if slices[i][0].start == 0 or slices[i][0].stop == h or slices[i][1].start == 0 or slices[i][1].stop == w:
                islands[i] = None

    # filter out tiny islands
    for i, island in enumerate(islands):
        if island is not None and island.sum() < min_island_area:
            islands[i] = None

    # zip the islands with their origins
    output = []
    for i, island in enumerate(islands):
        if island is not None:
            output.append((island, origins[i]))

    return output


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


def remove_stragglers(pixels):
    """
    Given a 2D array of binary pixels (already padded), removes any pixels that are "dangling"
    and only connected to one or two other pixels.
    Also fills "cracks" that are a hairline wide of 0s surrounded by at least 6 pixels of 1s.
    Requires the input image to be padded with 0s around the border
    Returns True if any modifications were made.
    """
    removed = False
    height, width = pixels.shape[0] - 2, pixels.shape[1] - 2  # Adjust for padding

    # Extract sub-arrays for each neighbor position (avoiding the added padding in calculations)
    neighbors = [pixels[y:y+height, x:x+width]
                 for y in range(3) for x in range(3) if not (y == 1 and x == 1)]

    # Sum the values of all neighbors
    neighbor_sum = sum(neighbors)

    # Middle part of pixels (avoiding padded edges) for processing
    core_pixels = pixels[1:-1, 1:-1]

    # Detect and remove stragglers (pixels with 1 or 2 neighbors)
    stragglers = (core_pixels == 1) & (neighbor_sum <= 2)
    if np.any(stragglers):
        core_pixels[stragglers] = 0
        removed = True

    # Detect and fill cracks (0s with 6 or more neighbors)
    cracks = (core_pixels == 0) & (neighbor_sum >= 6)
    if np.any(cracks):
        core_pixels[cracks] = 1
        removed = True

    # Remove peninsulas (the 3x3 center of a 5x5 slice, where only one border pixel is white (1)
    # and the center pixel is white). The 3x3 center will all be set to 0
    #
    # 1  0  0  0  0    0  0  0  0  0    0  0  0  0  0
    # 0  1  1  0  0    1  1  1  1  0    0  0  0  0  0
    # 0  1 (1) 0  0    0  1 (1) 1  0    0  0 (1) 0  0 ... etc
    # 0  0  0  0  0    0  1  0  0  0    0  1  0  0  0
    # 0  0  0  0  0    0  0  0  0  0    1  0  0  0  0

    # pad the pixels with a 2px black border for image processing so our sliding window doesn't
    # run beyond the allowable index
    padded_pixels = np.pad(pixels, pad_width=2, mode='constant', constant_values=0)
    height, width = pixels.shape

    # Compute the number of border pixels set to 1. Use padded_pixels so we don't index outside of the array
    borders = [padded_pixels[y:y+height, x:x+width]
               for y in range(5) for x in range(5) if not ((x == 1 or x == 2 or x == 3) and (y == 1 or y == 2 or y == 3))]
    border_sum = sum(borders)

    # Prepare an array of the same shape as pixels, where any peninsula pixels are represented by True.
    peninsulas = (pixels == 1) & (border_sum == 1)

    # Dilate that array to include the entire 3x3 center
    struct2 = ndimage.generate_binary_structure(2, 2) # generates a 3x3 matrix of True values
    peninsulas = ndimage.binary_dilation(peninsulas, structure=struct2)

    # Zero all of those peninsula pixels
    if np.any(peninsulas):
        pixels[peninsulas] = 0
        removed = True

    # Recurse if changes were made
    if removed:
        return remove_stragglers(pixels)

    return False
"""
A script that turns a bunch of overlapping photos into one large mosaic
Used for photographing the entire staging area

When ran directly on the command line, expects the data to already exist
Can also be invoked procedurally to build the mosiac as a robot takes photos
"""
import argparse
import os
import PIL
import yaml

from common import segment


class Mosaic(object):
    def __init__(self, pixels, w, h, origin):
        self.pixels = pixels
        self.w = w
        self.h = h

        # where was the robot at when this photo was taken
        self.origin = origin

        # what is the origin in the pixel array?
        # starts at (0, 0) but if the image grows up or left,
        # the origin would move to a positive index
        self.pixels_origin = (0, 0)

    def merge(self, pixels, w, h, at_point):
        print(f"The mosaic is currently {self.w}w x {self.h}h @ {self.pixels_origin}")
        print(f"\tMerging pixels ({w}x{h}) @ {at_point}")
        new_w = max(self.w, (at_point[0] + w)) - min(-self.pixels_origin[0], at_point[0])
        new_h = max(self.h, (at_point[1] + h)) - min(-self.pixels_origin[1], at_point[1])
        new_origin_x = -at_point[0] if at_point[0] < self.pixels_origin[0] else self.pixels_origin[0]
        new_origin_y = -at_point[1] if at_point[1] < self.pixels_origin[1] else self.pixels_origin[1]
        self.pixels = self._grow_pixels(self.pixels, self.w, self.h,
                                        left=new_origin_x, right=new_w - self.w,
                                        top=new_origin_y, bottom=new_h - self.h)
        self.w = new_w
        self.h = new_h
        self.pixels_origin = (new_origin_x, new_origin_y)
        print(f"\tThe mosaic is now {self.w}w x {self.h}h @ {self.pixels_origin}")

        # now overlay the new pixels
        # we take their values over the old values
        # TODO: maybe we should be smarter about this?
        # IDEA: if the old value is non-transparent, we could average?
        for y in range(h):
            for x in range(w):
                xi = x + at_point[0] + self.pixels_origin[0]
                yi = y + at_point[1] + self.pixels_origin[1]
                self.pixels[yi][xi] = pixels[y][x]

    def print(self):
        for y in range(self.h):
            for x in range(self.w):
                print(f"{self.pixels[y][x]}{self.pixels[y][x]}", end='')
            print()

    def save(self, output_path):
        img = PIL.Image.new('1', (self.w, self.h))
        img.putdata([pixel for row in self.pixels for pixel in row])
        img.save(output_path)

    @staticmethod
    def _grow_pixels(pixels, w, h, left, right, top, bottom):
        if left < 0 or right < 0 or top < 0 or bottom < 0:
            raise RuntimeError(f"{w}w x {h}h   <--{left}, {right}-->, ^{top}, v{bottom}")

        new_pixels = []

        # make all the new pixels transparent to start
        for y in range(h + top + bottom):
            new_pixels.append([2] * (w + left + right))

        # copy in all the old pixels into the correct location
        for y in range(h):
            for x in range(w):
                new_pixels[y + top][x + left] = pixels[y][x]
        return new_pixels


SSD_SEARCH_RADIUS = 0 # 5 is a value that is fast but still accounts for some slop
INPUT_UNITS_TO_PIXEL_RATIO = 1080
THRESHOLD = 170
SCALE_BY = 0.85


def grow(mosaic, new_path, photo_origin):
    (pixels, w, h) = segment.segment(filename=new_path, threshold=THRESHOLD, scale_by=SCALE_BY, clean_and_crop=False)
    print(f"Adding {new_path.split('/')[-1]} @ {photo_origin} ({w}x{h}) to mosaic...")
    minx, miny = None, None
    min_ssd = None
    for x in range(-SSD_SEARCH_RADIUS, SSD_SEARCH_RADIUS + 1):
        for y in range(-SSD_SEARCH_RADIUS, SSD_SEARCH_RADIUS + 1):
            xx = x + INPUT_UNITS_TO_PIXEL_RATIO * (photo_origin[0] - mosaic.origin[0]) + mosaic.pixels_origin[0]
            yy = y + INPUT_UNITS_TO_PIXEL_RATIO * (photo_origin[1] - mosaic.origin[1]) + mosaic.pixels_origin[1]
            xx = round(xx)
            yy = round(yy)
            print(xx, yy)
            ssd, scaled = _ssd(mosaic, pixels, w, h, xx, yy)
            if ssd is None:
                # no overlap
                continue
            if min_ssd is None or ssd < min_ssd:
                print(f"[{x}, {y}] => \t{ssd} \t{round(scaled * 5000)}")
                minx, miny, min_ssd = xx, yy, ssd

    # todo - merge and grow
    min_point = (minx, miny)
    mosaic.merge(pixels, w, h, at_point=min_point)


def _ssd(mosaic, pixels, w, h, x, y):
    ssd = 0
    count = 0
    for iy in range(h):
        for ix in range(w):
            jy = iy + y
            jx = ix + x
            if jy < 0 or jy >= len(mosaic.pixels) or jx < 0 or jx >= len(mosaic.pixels[0]):
                continue
            pval = pixels[iy][ix]
            mval = mosaic.pixels[jy][jx]

            if mval == 2:
                # the mosaic will have transparent areas as it grows outward
                # these are represented as pixel=2 and should also be considered out-of-bounds
                continue

            ssd += (pval - mval) ** 2
            count += 1
    return ssd, (ssd/count if count else 1000000)


def build_from_directory(directory_path, position_data, start_index=1, output_path=None):
    pixels, w, h = segment.segment(filename=os.path.join(directory_path, f"{start_index}.jpg"), threshold=THRESHOLD, scale_by=SCALE_BY, clean_and_crop=False)
    mosaic = Mosaic(pixels, w, h, origin=position_data[start_index])

    i = start_index + 1
    while os.path.exists(os.path.join(directory_path, f"{i}.jpg")) and i in position_data:
        new_path = os.path.join(directory_path, f"{i}.jpg")
        photo_origin = position_data[i]
        grow(mosaic, new_path, photo_origin)
        i += 1

    # mosaic.print()
    if output_path:
        mosaic.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory-path', required=True, help='Path to a directory full of overlapping images')
    parser.add_argument('--position-data-path', required=True, help='Path to a YAML file that contains the position each photo was taken at')
    parser.add_argument('--start-at-index', required=False, default=1, help='start processing from i.jpg')
    parser.add_argument('--output-path', required=False, default=None, help='Where to save the mosaic BMP to')
    args = parser.parse_args()

    with open(args.position_data_path) as f:
        position_data = yaml.safe_load(f)

    build_from_directory(args.directory_path, position_data, args.start_at_index, args.output_path)

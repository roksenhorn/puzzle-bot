# Currently unused but useful for debugging
import os
import os
import multiprocessing
import re
import PIL

from common import extract, util


SEGMENT_DIR = '2_segmented'


def dedupe_on_bmps(path):
    fill_islands(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, "2a_segmented"))
    thumbnail(input_path=os.path.join(path, "2a_segmented"), output_path=os.path.join(path, "2b_ssd"))
    ssd(input_path=os.path.join(path, "2b_ssd"), delete_path=os.path.join(path, SEGMENT_DIR))


def fill_islands(input_path, output_path):
    fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(_fill_islands, args)


def _fill_islands(args):
    input_path, output_path = args
    print(f"fill_islands({input_path}, {output_path})")
    pixels, width, height = util.load_bmp_as_binary_pixels(input_path)
    islands = util.find_islands(pixels, ignore_islands_along_border=True, island_value=0)

    print(f"Found {len(islands)} islands in {input_path.split('/')[-1]}")
    for i, island in enumerate(islands):
        for (x, y) in island:
            pixels[y][x] = 1

    img = PIL.Image.new('1', (width, height))
    img.putdata([pixel for row in pixels for pixel in row])
    img.save(output_path)


def thumbnail(input_path, output_path):
    fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    args = []
    for f in fs:
        # load the image and save off a thumbnail version scaled down to 20% then padded to 60x60
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(_thumbnail, args)


def _thumbnail(args):
    input_path, output_path = args
    img = PIL.Image.open(input_path)
    w, h = img.size[0] // 10, img.size[1] // 10
    img.thumbnail((w, h), PIL.Image.NEAREST)
    img = img.convert('1')
    img = img.crop((0, 0, 66, 66))
    img.save(output_path)
    print(f"Scaled down {input_path.split('/')[-1]} to {w}x{h}")


def ssd(input_path, delete_path):
    fs = [os.path.join(input_path, f) for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    images = [util.load_bmp_as_binary_pixels(f)[0] for f in fs]

    dupes = set()
    keeps = set()

    for i, img in enumerate(images):
        if i in dupes:
            continue

        for j, other_img in enumerate(images):
            if i == j:
                continue
            ssd_score = util.normalized_ssd(img, other_img)
            fi = fs[i].split('/')[-1].split('.')[0]
            fj = fs[j].split('/')[-1].split('.')[0]
            if ssd_score < 2.0:
                print(f"SSD between {fi} and {fj} is {ssd_score}")
                dupes.add(j)

        keeps.add(i)

    print(f"Starting with {len(images)} images")
    print(f"Removing {len(dupes)} images")
    print(f"Resulting in {len(keeps)} images")
    for j in dupes:
        f = fs[j]
        f = f.replace('2b_ssd', '2_segmented')
        os.remove(f)

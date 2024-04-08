# Currently unused but useful for debugging
import os
import os
import multiprocessing
import re

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
        pool.map(extract.fill_islands, args)


def thumbnail(input_path, output_path):
    fs = [f for f in os.listdir(input_path) if re.match(r'.*\.bmp', f)]
    args = []
    for f in fs:
        # load the image and save off a thumbnail version scaled down to 20% then padded to 60x60
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f)
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(extract.thumbnail, args)


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

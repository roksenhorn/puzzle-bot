import os
from PIL import Image
import numpy as np

from common import util
import shutil


def deduplicate(input_path, output_path):
    """
    Removes duplicate pieces from the path
    Duplicates are defined as two pieces with strikingly similar pixel data
    """

    # TODO:
    # Use the piece location to make sure we're only removing pieces that are physically at the same location

    tiny_pieces = {}
    for f in os.listdir(input_path):
        f = os.path.join(input_path, f)
        with Image.open(f) as img:
            w, h = img.size
            img = img.resize((w // 5, h // 5), resample=Image.NEAREST)

            # pad all to the same size
            img = img.crop((0, 0, 70, 70))

            tiny_pieces[f] = np.array(img.getdata())

    # tuples of matching file paths
    dupes = set()
    visited = set()

    # compare each piece to every other piece using SSD
    for f1, p1 in tiny_pieces.items():
        for f2, p2 in tiny_pieces.items():
            if f1 == f2 or (f1, f2) in visited:
                continue

            visited.add((f1, f2))
            visited.add((f2, f1))

            ssd = util.normalized_ssd(p1, p2)
            if ssd < 2600:
                if ssd > 2000:
                    print(f"Close dupes that make the cut: {f1} <> {f2}: \t {ssd}")

                if f1 < f2:
                    dupes.add(f1)
                else:
                    dupes.add(f2)
            elif ssd < 3600:
                print(f"\t Close dupes that missed the cut: {f1} <> {f2}: \t {ssd}")

    uniques = set(tiny_pieces.keys()) - dupes
    print(f"Found {len(dupes)} duplicate pieces, resulting in {len(uniques)} unique pieces.")

    # finally, save off all the images, sequentially numbered
    for id, f in enumerate(uniques):
        # copy the file to the output directory
        out = os.path.join(output_path, f'{id + 1}.bmp')
        shutil.copyfile(f, out)

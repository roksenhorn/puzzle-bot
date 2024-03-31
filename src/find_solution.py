import cProfile
import argparse
import os
import time
import multiprocessing
import re
import PIL

from common import bmp, board, connect, dedupe, extract, util, vector


# Step 1 takes in photos of pieces on the bed and output BMPs that contain multiple pieces
PHOTOS_DIR = '0_photos'
PHOTO_BMP_DIR = '1_photo_bmps'

# Step 2 takes in photo BMPs and outputs cleaned up individual pieces as bitmaps
SEGMENT_DIR = '2_segmented'

# Step 3 goes through all the BMP pieces and deletes duplicates
DEDUPED_DIR = '3_deduped'

# Step 4 takes in piece BMPs and outputs SVGs
VECTOR_DIR = '4_vector'

# Step 5 takes in SVGs and outputs a graph of connectivity
CONNECTIVITY_DIR = '5_connectivity'

# Step 6 takes in the graph of connectivity and outputs a solution
SOLUTION_DIR = '6_solution'


def solve(path, serialize, id, start_at_step, stop_before_step):
    start_time = time.time()

    for i, d in enumerate([PHOTOS_DIR, PHOTO_BMP_DIR, SEGMENT_DIR, DEDUPED_DIR, VECTOR_DIR, CONNECTIVITY_DIR, SOLUTION_DIR]):
        os.makedirs(os.path.join(path, d), exist_ok=True)

        # wipe any directories we'll act on, except 0 which is the input
        if i != 0 and i > start_at_step and i <= stop_before_step:
            for f in os.listdir(os.path.join(path, d)):
                os.remove(os.path.join(path, d, f))

    if start_at_step <= 0 and stop_before_step > 0:
        bmp_all(input_path=os.path.join(path, PHOTOS_DIR), output_path=os.path.join(path, PHOTO_BMP_DIR), id=id)

    if start_at_step <= 1 and stop_before_step > 1:
        extract_all(input_path=os.path.join(path, PHOTO_BMP_DIR), output_path=os.path.join(path, SEGMENT_DIR), id=id)

    if start_at_step <= 2 and stop_before_step > 2:
        deduplicate(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, DEDUPED_DIR))

    if start_at_step <= 3 and stop_before_step > 3:
        vectorize(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, VECTOR_DIR), id=id, serialize=serialize)

    if start_at_step <= 4 and stop_before_step > 4:
        find_connectivity(input_path=os.path.join(path, VECTOR_DIR), output_path=os.path.join(path, CONNECTIVITY_DIR), id=id, serialize=serialize)

    if start_at_step <= 5 and stop_before_step > 5 and not id:
        build_board(input_path=os.path.join(path, CONNECTIVITY_DIR), output_path=os.path.join(path, SOLUTION_DIR))

    if stop_before_step is None:
        duration = time.time() - start_time
        print(f"\n\n{util.GREEN}### Puzzle solved in {round(duration, 2)} sec ###{util.WHITE}\n")


def bmp_all(input_path, output_path, id):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    if id:
        fs = [f'{id}.jpeg']
    else:
        fs = [f for f in os.listdir(input_path) if re.match(r'.*\.jpe?g', f)]
        id = 1

    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path, f'{id}.bmp')
        args.append([input_img_path, output_img_path])
        id += 1

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(bmp.photo_to_bmp, args)


def extract_all(input_path, output_path, id):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    if id:
        fs = [f'{id}.bmp']
    else:
        fs = [f for f in os.listdir(input_path) if re.match(r'[0-9]+\.bmp', f)]
        id = 1

    args = []
    for f in fs:
        input_img_path = os.path.join(input_path, f)
        output_img_path = os.path.join(output_path)
        unique_id = id
        args.append([input_img_path, output_img_path, unique_id])
        id += 1

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(extract.extract_pieces, args)


def deduplicate(input_path, output_path):
    # deduplicate the pieces - often times the same piece was successfully extracted from multiple photos
    print(f"\n{util.RED}### Deduplicated extracted pieces ###{util.WHITE}\n")
    dedupe.deduplicate(input_path, output_path)


def vectorize(input_path, output_path, id, serialize):
    """
    Loads each image.bmp in the input directory, converts it to an SVG in the output directory
    """
    print(f"\n{util.RED}### Vectorizing ###{util.WHITE}\n")

    start_time = time.time()
    i = id if id is not None else 1

    args = []
    while os.path.exists(os.path.join(input_path, f'{i}.bmp')):
        path = os.path.join(input_path, f'{i}.bmp')
        render = (i == id)
        args.append([path, i, output_path, render])

        i += 1
        if id is not None:
            break

    if serialize:
        for arg in args:
            vector.load_and_vectorize(arg)
    else:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.map(vector.load_and_vectorize, args)

    duration = time.time() - start_time
    print(f"Vectorizing took {round(duration, 2)} seconds ({round(duration /i, 2)} seconds per piece)")


def find_connectivity(input_path, output_path, id, serialize):
    """
    Opens each piece data and finds how each piece could connect to others
    """
    print(f"\n{util.RED}### Building connectivity ###{util.WHITE}\n")
    start_time = time.time()
    connect.build(input_path, output_path, id, serialize)
    duration = time.time() - start_time
    print(f"Building the graph took {round(duration, 2)} seconds")


def build_board(input_path, output_path):
    """
    Searches connectivity to find the solution
    """
    print(f"\n{util.RED}### Build board ###{util.WHITE}\n")
    start_time = time.time()
    board.build(input_path, output_path)
    duration = time.time() - start_time
    print(f"Building the board took {round(duration, 2)} seconds")


def rename(path):
    d = os.path.join(path, PHOTOS_DIR)

    # for each file in this directory, rename it to i.jpeg
    i = 1

    # only rename jpg or jpeg
    previous_files = [f for f in os.listdir(d) if re.match(r'.*\.jpe?g', f)]
    print(f"There are {len(previous_files)} files to rename")

    for f in previous_files:
        src = os.path.join(d, f)
        dest = os.path.join(d, f'{i}.jpeg.tmp')
        os.rename(src, dest)
        i += 1

    i = 1
    for f in previous_files:
        src = os.path.join(d, f'{i}.jpeg.tmp')
        dest = os.path.join(d, f'{i}.jpeg')
        os.rename(src, dest)
        i += 1
    print(f"Renamed {len(previous_files)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the base directory that has a dir `0_input` full of JPEGs in it')
    parser.add_argument('--only-process-id', default=None, required=False, help='Only processes the provided ID', type=int)
    parser.add_argument('--start-at-step', default=0, required=False, help='Start processing at this step', type=int)
    parser.add_argument('--stop-before-step', default=10, required=False, help='Stop processing at this step', type=int)
    parser.add_argument('--serialize', default=False, action="store_true", help='Single-thread processing')
    parser.add_argument('--rename', default=False, action="store_true", help='Renames the input files to 1.jpeg, 2.jpeg, ...')
    args = parser.parse_args()

    if args.rename:
        rename(path=args.path)

    solve(path=args.path, serialize=args.serialize, id=args.only_process_id, start_at_step=args.start_at_step, stop_before_step=args.stop_before_step)


if __name__ == '__main__':
    PROFILE = False
    if PROFILE:
        cProfile.run('main()', 'profile_results.prof')
    else:
        main()

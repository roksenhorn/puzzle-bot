import argparse
import os
import yaml
import time
import threading

import connect
import segment
import vector
import util
import board


INPUT_DIR = '0_input'
SEGMENT_DIR = '1_segmented'
VECTOR_DIR = '2_vector'
CONNECTIVITY_DIR = '3_connectivity'
SOLUTION_DIR = '4_solution'


def solve(path, id, skip_step):
    start_time = time.time()

    for d in [INPUT_DIR, SEGMENT_DIR, VECTOR_DIR, CONNECTIVITY_DIR, SOLUTION_DIR]:
        os.makedirs(os.path.join(path, d), exist_ok=True)

    if skip_step < 0:
        segment_each(input_path=os.path.join(path, INPUT_DIR), output_path=os.path.join(path, SEGMENT_DIR), id=id)

    if skip_step < 1:
        vectorize(input_path=os.path.join(path, SEGMENT_DIR), output_path=os.path.join(path, VECTOR_DIR), id=id)

    if skip_step < 2:
        find_connectivity(input_path=os.path.join(path, VECTOR_DIR), output_path=os.path.join(path, CONNECTIVITY_DIR), id=id)

    if skip_step < 3:
        build_board(input_path=os.path.join(path, CONNECTIVITY_DIR), output_path=os.path.join(path, SOLUTION_DIR))

    print('')

    duration = time.time() - start_time
    print(f"\n{util.GREEN}### Puzzle solved in {round(duration, 2)} sec ###{util.WHITE}\n")


def segment_each(input_path, output_path, id):
    """
    Loads each image in the input directory, segments it into a black & white bitmap in the output directory
    """
    print(f"\n{util.RED}### Segmenting ###{util.WHITE}\n")

    start_time = time.time()
    i = id if id is not None else 1
    thrs = []
    while os.path.exists(os.path.join(input_path, f'{i}.jpeg')):
        input_img_path = os.path.join(input_path, f'{i}.jpeg')
        output_img_path = os.path.join(output_path, f'{i}.bmp')

        print(f"> Segmenting image {input_img_path} into {output_img_path}")
        thr = threading.Thread(target=segment.segment, args=(input_img_path, output_img_path))
        thr.start()
        thrs.append(thr)

        if len(thrs) > 20:
            for thr in thrs:
                thr.join()
            thrs = []

        i += 1
        if id is not None:
            break

    for thr in thrs:
        thr.join()

    duration = time.time() - start_time
    print(f"Segmenting took {round(duration, 2)} seconds ({round(duration/i, 2)} seconds per image)")


def vectorize(input_path, output_path, id):
    """
    Loads each image.bmp in the input directory, converts it to an SVG in the output directory
    """
    print(f"\n{util.RED}### Vectorizing ###{util.WHITE}\n")

    start_time = time.time()
    i = id if id is not None else 1
    thrs = []
    parallelize = True
    while os.path.exists(os.path.join(input_path, f'{i}.bmp')):
        print(f"> Vectorizing {i}.bmp")
        path = os.path.join(input_path, f'{i}.bmp')
        vectorize = vector.Vector.from_file(filename=path, id=i)
        render = (i == id)
        thr = threading.Thread(target=vectorize.process, args=(output_path, render))
        thr.start()
        thrs.append(thr)

        if len(thrs) > 8 or not parallelize:
            for thr in thrs:
                thr.join()
            thrs = []

        i += 1
        if id is not None:
            break

    # for thr in thrs:
    #     thr.join()

    duration = time.time() - start_time
    print(f"Vectorizing took {round(duration, 2)} seconds ({round(duration /i, 2)} seconds per piece)")


def find_connectivity(input_path, output_path, id):
    """
    Opens each piece data and finds how each piece could connect to others
    """
    print(f"\n{util.RED}### Building connectivity ###{util.WHITE}\n")

    start_time = time.time()
    connect.build(input_path, output_path, id)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the base directory that has a dir `0_input` full of JPEGs in it')
    parser.add_argument('--only-process-id', default=None, required=False, help='Only processes the provided ID', type=int)
    parser.add_argument('--skip-step', default=-1, required=False, help='Start processing from after this step', type=int)
    args = parser.parse_args()

    solve(path=args.path, id=args.only_process_id, skip_step=args.skip_step)
# jigsaw-bot
A python jigsaw puzzle solver

## Overview

This solver is designed to run in two modes: 

- *Functional solver* where a directory of puzzle piece photos are passed in once, and a solution is computed and spit out
- *Solving service* where imagery can be shared as it is collected, and the puzzle will be progressively solved, then serve up a final solution

## Usage: Functional Solver

1. Drop images into the `./0_input` directory, named starting from `1.jpeg`, onward. There is no significance to the ordering of the numbers.
2. `python3 solve.py --path root/of/repo
3. To skip processing steps, you can optionally provide the argument `--skip n` to start from step `n+1`
4. To only process one piece, you can provide the argument `--only i`

## Usage: Solving Service

This mode isn't implemented yet, actually.

## Utilities

### Idnetifying a piece

If you have a new photo of an existing piece and need to be able to identify it, you can use the following command:

```
python3 find.py --photo-path path/to/new/photo.jpg --puzzle-dir .
```

## Limitations

- The solver will not determine the dimensions of your puzzle. It is hardcoded (e.g. 10x10). This can be changed in `board.py`.
- Photos must be taken on a bright background. Segmentation removes all near-white pixels.
- Currently only sovles puzzles where pieces are four-sides with somewhat normal geometries.
- It isn't perfect and doesn't have great error handling, but the fundamentals are working well.

## TODOs

- The puzzle dimensions are hardcoded. This is obviously cheating.
- Running as a service
- Saving the solution to disk
- Identified piece needs to return orientation too
- Specify a solution file for better debugging logs as it attempts to solve
- requirements.txt


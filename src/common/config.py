"""
Common configuration for the puzzle bot
"""

# Paramaters for photo segmentation
BMP_WIDTH = 2100  # scale the BMP to this wide
CROP_TRBL = (0, 0, 0, 0)  # crop the post-scaled photo by this many pixels on the top, right, bottom, left
MIN_PIECE_AREA = 200*200
SEG_THRESH = 190  # for white pieces, raise this to cut tighter into the border

# dimensions for the puzzle you're solving
PUZZLE_WIDTH = 40
PUZZLE_HEIGHT = 25
PUZZLE_NUM_PIECES = PUZZLE_WIDTH * PUZZLE_HEIGHT
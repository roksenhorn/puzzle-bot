"""
Common configuration for the puzzle bot
"""

# Paramaters for photo segmentation
BMP_WIDTH = None  # 2100  # scale the BMP to this wide or None to turn off scaling
MIN_PIECE_AREA = 200*200
SEG_THRESH = 135  # for white pieces, raise this to cut tighter into the border (ICC changed from 190 to 180 as an experiment)


# dimensions for the puzzle you're solving
PUZZLE_WIDTH = 40
PUZZLE_HEIGHT = 25
PUZZLE_NUM_PIECES = PUZZLE_WIDTH * PUZZLE_HEIGHT


# Directory structure for data processing
# Step 1 takes in photos of pieces on the bed and outputs binary BMPs of those photos
PHOTOS_DIR = '0_photos'
PHOTO_BMP_DIR = '1_photo_bmps'

# Step 2 takes in photo BMPs and outputs cleaned up individual pieces as bitmaps
SEGMENT_DIR = '2_segmented'

# Step 4 takes in piece BMPs and outputs SVGs
VECTOR_DIR = '3_vector'

# Step 4 goes through all the vector pieces and deletes duplicates
DEDUPED_DIR = '4_deduped'

# Step 5 takes in SVGs and outputs a graph of connectivity
CONNECTIVITY_DIR = '5_connectivity'

# Step 6 takes in the graph of connectivity and outputs a solution
SOLUTION_DIR = '6_solution'

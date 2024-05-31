"""
Common configuration for the puzzle bot
"""

# Paramaters for photo segmentation
SCALE_BMP_TO_WIDTH = None  # scale the BMP to this wide or None to turn off scaling
CROP_TOP_RIGHT_BOTTOM_LEFT = (620, 860, 620, 860)  # crop the BMP by this many pixels on each side
MIN_PIECE_AREA = 200*200
SEG_THRESH = 130  # raise this to cut tighter into the border


# dimensions for the puzzle you're solving
PUZZLE_WIDTH = 40
PUZZLE_HEIGHT = 25
PUZZLE_NUM_PIECES = PUZZLE_WIDTH * PUZZLE_HEIGHT
TIGHTEN_RELAX_PX = 3.5  # positive = add space between pieces, negative = remove space between pieces


# Robot parameters
APPROX_ROBOT_COUNTS_PER_PIXEL = 10


# Directory structure for data processing
# Step 1 takes in photos of pieces on the bed and outputs binary BMPs of those photos
PHOTOS_DIR = '0_photos'
PHOTO_BMP_DIR = '1_photo_bmps'

# Step 2 takes in photo BMPs and outputs cleaned up individual pieces as bitmaps
SEGMENT_DIR = '2_segmented'

# Step 3 goes through all the vector pieces and deletes duplicates
DEDUPED_DIR = '3_deduped'

# Step 4 takes in piece BMPs and outputs SVGs
VECTOR_DIR = '4_vector'

# Step 5 takes in SVGs and outputs a graph of connectivity
CONNECTIVITY_DIR = '5_connectivity'

# Step 6 takes in the graph of connectivity and outputs a solution
SOLUTION_DIR = '6_solution'

# Step 7 adjusts the tightness of the solved puzzle: how much breathing room do pieces need to actually click together?
TIGHTNESS_DIR = '7_tightness'

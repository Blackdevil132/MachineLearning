# color shortcuts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR_BG = (230, 230, 230)

# subdirectory for qtables
PATH = "qtables/"

# Environment defines
NUM_ACTIONS = 6
NUM_ROWS = 8
NUM_COLS = 8
STATE_DEAD = 64
NUM_STATES = NUM_ROWS * NUM_COLS

# Learning hyperparameters
NUM_EPISODES = 500000
LEARNING_RATE = 0.5
DISCOUNT_RATE = 0.9
DECAY_RATE = 0.0002

# Learning parameters
INIT_STATE = bytes((0, NUM_COLS-1, (NUM_ROWS-1)*NUM_COLS))
MAX_STEPS = 40  # Max steps per episode
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
MAP_NAME = "8x8"

# Memory parameters
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
PRETRAIN_SIZE = 4

"""
F:  Normal Tile
H:  Hole, ends episode
G:  Goal, ends episode
S:  Starting Tile
K:  Killreward
P:  Penalty, applied for stepping out of Bounds or slaying without reason
"""
REWARDS = {b'F': 0, b'H': -100, b'G': 100, b'S': 0, b'K': 50, b'P': -100}

# customisation of output of main.py
SHOW_QTABLE = False         # print out qtable after learning
SHOW_MAP = False            # print maplayout after learning

NUM_TESTS = 100             # number of test to determine statistics after learning
SHOW_TESTS = True           # Show paths of test games
SHOW_ONLY_SUBPAR = True     # only print path of tests with results below THRESHOLD
THRESHOLD = 200             # Threshold for subpar tests

# Action assignment to numbers
STAY = 0
LEFT = 1
DOWN = 2
RIGHT = 3
UP = 4
SLAY = 5

IntToAction = {0: "STAYING", 1: "LEFT", 2: "DOWN", 3: "RIGHT", 4: "UP", 5: "SLAYING"}

# Map layout
MAP = bytes("FFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG", "utf8")

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

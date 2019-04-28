LEFT = 1
DOWN = 2
RIGHT = 3
UP = 4
STAY = 0
SLAY = 5

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

IntToAction = ["STAYING", "LEFT", "DOWN", "RIGHT", "UP", "SLAYING"]

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR_BG = (230, 230, 230)

PATH = "qtables/"

MAP = bytes("FFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG", "utf8")
MAX_STEPS = 35  # Max steps per episode

NUM_ACTIONS = 6
NUM_ROWS = 8
NUM_COLS = 8

NUM_STATES = NUM_ROWS * NUM_COLS

NUM_TESTS = 100
MAP_NAME = "8x8"
SHOW_QTABLE = False
SHOW_TESTS = False

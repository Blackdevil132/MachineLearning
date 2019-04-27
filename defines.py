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
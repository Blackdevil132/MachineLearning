import sys

from src.qrl.QRL import QRL
from src.environments.Game2Enemies import Game2Enemies
from defines import *

if len(sys.argv) < 2:
    # initialize standard parameters
    qtable = PATH + "190427_18"
else:
    # initialize with given parameter-values
    qtable = sys.argv[1]


env = Game2Enemies(map_name=MAP_NAME)

qrl = QRL(env)
qrl.loadFromFile(qtable)
qrl.test_visual()

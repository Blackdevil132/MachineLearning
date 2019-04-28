import pickle, os
from src.GameEnemy import GameEnemy
from src.QRL import QRL
from src.Game2Enemies import Game2Enemies
from src.tools.helpers import bytes2long, long2bytes
from src.tools.tools import timeit
import numpy as np
from defines import *

qrl = QRL(Game2Enemies(map_name="8x8"))
qrl.loadFromFile("qtables/190427_18")
qrl.test_visual()

from src.GameEnemy import GameEnemy
from src.QRL import QRL
from src.Game2Enemies import Game2Enemies
import numpy as np
from defines import *

qrl = QRL(Game2Enemies(map_name="8x8"))
qrl.loadFromFile("qtables/190427_18")
qrl.test_visual()

"""
done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()"""

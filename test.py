import pickle, os
from src.GameEnemy import GameEnemy
from src.QRL import QRL
from src.Game2Enemies import Game2Enemies
from src.tools.helpers import bytes2long, long2bytes
from src.tools.tools import timeit
import numpy as np
from defines import *

#qrl = QRL(Game2Enemies(map_name="8x8"))
#qrl.loadFromFile("qtables/190427_18")
#qrl.test_visual()

"""
done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()"""
"""
env = Game2Enemies(map_name="8x8")"""

#transitions = {bytes((0, 7, 56)): {0: [(0.33, bytes((1, 7, 56)), 100, False), (0.33, bytes((8, 7, 56)), -100, False), (0.33, bytes((0, 7, 48)), 0, True)], 1: [(1.0, bytes((0, 7, 56)), 100, True)]}}
with open("transitions.pkl", 'rb') as f:
    print("Loading Transition Matrix from %s..." % "transitions.pkl", end='')
    transitions = pickle.load(f)
    print("done")


timeit(save, [transitions])
t1 = timeit(load_o, [])

print("OK")
limit = 5
counter = 0
for s in t1.keys():
    for a in transitions[s]:
        if transitions[s][a] != t1[s][a]:
            print(s, a)
            print(transitions[s][a])
            print(t1[s][a])
            break

print(counter)

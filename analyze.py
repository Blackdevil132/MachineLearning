import pickle
import numpy as np
from src.Game2Enemies import Game2Enemies
from src.tools.tools import timeit

env = timeit(Game2Enemies, [None, "8x8"])

with open("transitions.pkl", 'rb') as file:
    print("Trying to load Transition Matrix from %s..." % "transitions.pkl")
    P = pickle.load(file)

for s in P.keys():
    if s == bytes((40, 64, 48)):
        for a in P[s].keys():
            print(P[s][a])

import pickle
import numpy as np
from src.Game2Enemies import Game2Enemies
from src.tools.tools import timeit

#env = timeit(Game2Enemies, [None, "8x8"])

with open("transitions.pkl", 'rb') as file:
    print("Trying to load Transition Matrix from %s..." % "transitions.pkl")
    P = pickle.load(file)

lens = np.zeros(37)
for s in P.keys():
    for a in P[s].keys():
        lens[len(P[s][a])] += 1

print(lens)
print(sum(list(map(lambda x: x*6, lens))))

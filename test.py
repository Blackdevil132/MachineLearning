import pickle, os
from src.GameEnemy import GameEnemy
from src.QRL import QRL
from src.Game2Enemies import Game2Enemies
from src.tools.helpers import bytes2long, long2bytes
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


#transitions = {bytes((0, 7, 56)): {0: [(0.33, bytes((1, 7, 56)), 100, False), (0.33, bytes((8, 7, 56)), -100, False), (0.33, bytes((0, 7, 48)), 0, True)], 1: [(1.0, bytes((0, 7, 56)), 100, True)]}}
with open("transitions.pkl", 'rb') as file:
    print("Loading Transition Matrix from %s..." % "transitions.pkl", end='')
    transitions = pickle.load(file)
    print("done")

print("Writing to archive...", end='')
offset = [0 for i in range(37)]
os.makedirs("transitions", exist_ok=True)
with open("transitions/index.bin", 'wb') as index:
    for s in transitions.keys():
        for a in transitions[s]:
            len_t = len(transitions[s][a])
            index_entry = long2bytes(offset[len_t], 1, 4)
            index_entry[0] = len_t
            index.write(index_entry)
            with open("transitions/%i" % len_t, 'ab') as file:
                file.seek(offset[len_t])
                for t in transitions[s][a]:
                    p, ns, r, d = t
                    #print(p, ns, r, d)
                    #print(type(p), type(ns), type(r), type(d))
                    row_bytes = bytes((round(100*float(p)), *[i for i in ns], r % (1 << 8), d))
                    file.write(row_bytes)
                    offset[len_t] += 6

print("complete")

print("Loading from archive...", end='')
t1 = {}
with open("transitions/index.bin", 'rb') as index:
    for s1 in range(64):
        for s2 in range(65):
            for s3 in range(65):
                s = bytes((s1, s2, s3))
                t1[s] = {}
                for a in range(6):
                    t1[s][a] = []
                    index_entry = index.read(5)
                    if index_entry == b'':
                        break
                    file_id = bytes2long(index_entry, 0, 1)
                    file_offset = bytes2long(index_entry, 1, 4)
                    print(file_id, file_offset)
                    with open("transitions/%i" % file_id, 'rb') as file:
                        file.seek(file_offset)
                        for i in range(file_id):
                            inp = file.read(6)
                            if inp == b'':
                                break
                            t1[s][a].append((inp[0]/100.0, inp[1:4], inp[4] if inp[4] < 128 else inp[4] % -(1 << 8), bool(inp[5])))

print("done")

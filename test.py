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


transitions = {bytes((0, 7, 56)): {0: [(0.33, bytes((1, 7, 56)), 100, False), (0.33, bytes((8, 7, 56)), -100, False), (0.33, bytes((0, 7, 48)), 0, True)], 1: [(1.0, bytes((0, 7, 56)), 100, True)]}}

os.makedirs("transitions", exist_ok=True)
with open("transitions/index.bin", 'wb') as index:
    for s in transitions.keys():
        for a in transitions[s]:
            os.makedirs("transitions/%i" % (s[0]), exist_ok=True)
            with open("transitions/%i/%i" % (s[0], a), 'wb') as file:
                offset = 0
                for t in transitions[s][a]:
                    p, ns, r, d = t
                    row_bytes = bytes((round(100*p), *[i for i in ns], r % (1 << 8), d))
                    file.write(row_bytes)
                    offset += 6
                index.write(long2bytes(offset))


t1 = {bytes((0, 7, 56)): {0: [], 1: []}}
s = bytes((0, 7, 56))
for a in [0, 1]:
    with open("transitions/%i/%i" % (s[0], a), 'rb') as file:
        while True:
            inp = file.read(6)
            if inp == b'':
                break
            t1[s][a].append((inp[0]/100.0, inp[1:4], inp[4] if inp[4] < 128 else inp[4] % -(1 << 8), bool(inp[5])))

print(t1 == transitions)

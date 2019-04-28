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
env = Game2Enemies(map_name="8x8")

#transitions = {bytes((0, 7, 56)): {0: [(0.33, bytes((1, 7, 56)), 100, False), (0.33, bytes((8, 7, 56)), -100, False), (0.33, bytes((0, 7, 48)), 0, True)], 1: [(1.0, bytes((0, 7, 56)), 100, True)]}}
with open("transitions.pkl", 'rb') as f:
    print("Loading Transition Matrix from %s..." % "transitions.pkl", end='')
    transitions = pickle.load(f)
    print("done")"""


def save(transitions):
    print("Constructing archive files...")
    offset = [0 for i in range(37)]
    os.makedirs("transitions", exist_ok=True)
    index_cached = bytearray()
    files_cached = [bytearray() for i in range(37)]
    for s in transitions.keys():
        for a in transitions[s]:
            len_t = len(transitions[s][a])
            index_entry = long2bytes(offset[len_t], 1, 4)
            index_entry[0] = len_t
            index_cached += index_entry
            for t in transitions[s][a]:
                p, ns, r, d = t
                row_bytes = bytes((round(100*float(p)), *[i for i in ns], r % (1 << 8), d))
                files_cached[len_t] += row_bytes
                offset[len_t] += 6

    print("Writing archives to disk...", end='')
    for i in range(37):
        if files_cached[i] != b'':
            with open("transitions/%i" % i, 'ab') as file:
                file.write(files_cached[i])
    print("done")

    with open("transitions/index.bin", 'wb') as index:
        print("Writing index to disk...", end='')
        index.write(index_cached)

    print("complete")


def load():
    print("Loading from archive...", end='')
    transitions = {}
    with open("transitions/index.bin", 'rb') as index:
        for s1 in range(64):
            for s2 in range(65):
                for s3 in range(65):
                    s = bytes((s1, s2, s3))
                    transitions[s] = {}
                    for a in range(6):
                        transitions[s][a] = []
                        index_entry = index.read(5)
                        if index_entry == b'':
                            break
                        file_id = bytes2long(index_entry, 0, 1)
                        file_offset = bytes2long(index_entry, 1, 4)
                        #print(file_id, file_offset)
                        with open("transitions/%i" % file_id, 'rb') as file:
                            file.seek(file_offset)
                            for i in range(file_id):
                                inp = file.read(6)
                                if inp == b'':
                                    break
                                transitions[s][a].append((inp[0]/100.0, inp[1:4], inp[4] if inp[4] < 128 else inp[4] % -(1 << 8), bool(inp[5])))

    print("done")
    return transitions


def load_o():
    print("Loading Index...", end='')
    transitions = {}
    with open("transitions/index.bin", 'rb') as index:
        index_cached = index.read()
        index_offset = 0
    print("done")

    print("Loading archive files...", end='')
    files_cached = [b'' for i in range(37)]
    for file_id in range(37):
        try:
            with open("transitions/%i" % file_id, 'rb') as file:
                files_cached[file_id] = file.read()
        except FileNotFoundError:
            pass
    print("done")

    print("Reconstructing Matrix...", end='')
    for s1 in range(64):
        for s2 in range(65):
            for s3 in range(65):
                s = bytes((s1, s2, s3))
                transitions[s] = {}
                for a in range(6):
                    transitions[s][a] = []
                    index_entry = index_cached[index_offset:index_offset+5]
                    if index_entry == b'':
                        break
                    file_id = bytes2long(index_entry, 0, 1)
                    file_offset = bytes2long(index_entry, 1, 4)
                    row = files_cached[file_id][file_offset:file_offset+(6*file_id)]
                    for i in range(file_id):
                        inp = row[i*6:(i+1)*6]
                        if inp == b'':
                            break
                        transitions[s][a].append((inp[0]/100.0, inp[1:4], inp[4] if inp[4] < 128 else inp[4] % -(1 << 8), bool(inp[5])))

                    index_offset += 5

    print("done")
    return transitions


#timeit(save, [transitions])
t1 = timeit(load_o, [])

print("OK")
"""
for s in t1.keys():
    for a in t1[s]:
        print(s, a, t1[s][a])
"""
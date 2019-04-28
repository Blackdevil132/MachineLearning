from defines import *
import os


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(arr, r=0, c=0):
        if arr[r][c] == 'G':
            return True

        tmp = arr[r][c]
        arr[r][c] = "#"

        # Recursively check in all four directions.
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                continue

            if arr[r_new][c_new] not in '#H':
                if is_valid(arr, r_new, c_new):
                    arr[r][c] = tmp
                    return True

        arr[r][c] = tmp
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


def stepToString(step):
    output = []
    if step[1] == STAY:
        output.append("STAYING at %i." % step[2][0])
    elif step[1] == SLAY:
        if step[2][1] == 64:
            output.append("SLAYING Enemy at %i." % step[0][1])
        elif step[2][2] == 64:
            output.append("SLAYING Enemy at %i." % step[0][2])
        else:
            output.append("SLAYED Nothing.")
    else:
        output.append("Moving from %i %s to %i." % (step[0][0], IntToAction[step[1]], step[2][0]))

    if step[2][1] == 64:
        output.append("Enemy 1 is DEAD.")
    elif step[0][1] == step[2][1]:
        output.append("Enemy 1 is STAYING at %i." % step[2][1])
    else:
        output.append("Enemy 1 moving from %i to %i." % (step[0][1], step[2][1]))

    if step[2][2] == 64:
        output.append("Enemy 2 is DEAD.")
    elif step[0][2] == step[2][2]:
        output.append("Enemy 2 is STAYING at %i." % step[2][2])
    else:
        output.append("Enemy 2 moving from %i to %i." % (step[0][2], step[2][2]))
    output.append("Reward: %i." % step[3])
    return "{: <30} {: <30} {: <30} {: <20}".format(*output)


def bytes2long(bs, startIndex=0, length=8):
    longValue = 0
    for i in range(startIndex, startIndex + length):
        longValue = longValue << 8
        longValue |= (bs[i] & 0x000000ff)
    return longValue


def long2bytes(value, startIndex=0, length=8):
    destinationArray = [0 for i in range(length+startIndex)]
    for i in range((startIndex + length) - 1, startIndex - 1, -1):
        destinationArray[i] = value & 0x000000ff
        value = value >> 8

    return bytearray(destinationArray)


BYTES_FIELD = 5
BYTES_INDEX = 5
NUM_FILES = NUM_ACTIONS * NUM_ACTIONS + 1


def saveTransitions(transitions):
    print("Constructing archive files...")
    offset = [0 for i in range(NUM_FILES)]
    os.makedirs("transitions", exist_ok=True)
    index_cached = bytearray()
    files_cached = [bytearray() for i in range(NUM_FILES)]
    for s in transitions.keys():
        for a in transitions[s]:
            len_t = len(transitions[s][a])
            index_entry = long2bytes(offset[len_t], 1, BYTES_INDEX-1)
            index_entry[0] = len_t
            index_cached += index_entry
            for t in transitions[s][a]:
                p, ns, r, d = t
                row_bytes = bytes((*[i for i in ns], r % (1 << 8), d))
                files_cached[len_t] += row_bytes
                offset[len_t] += BYTES_FIELD

    print("Writing archives to disk...", end='')
    for i in range(NUM_FILES):
        if files_cached[i] != b'':
            with open("transitions/%i" % i, 'ab') as file:
                file.write(files_cached[i])
    print("done")

    with open("transitions/index.bin", 'wb') as index:
        print("Writing index to disk...", end='')
        index.write(index_cached)

    print("complete")


def loadTransitions():
    print("Loading Index...", end='')
    transitions = {}
    with open("transitions/index.bin", 'rb') as index:
        index_cached = index.read()
        index_offset = 0
    print("done")

    print("Loading archive files...", end='')
    files_cached = [b'' for i in range(NUM_FILES)]
    for file_id in range(NUM_FILES):
        try:
            with open("transitions/%i" % file_id, 'rb') as file:
                files_cached[file_id] = file.read()
        except FileNotFoundError:
            pass
    print("done")

    print("Reconstructing Matrix...", end='')
    for s1 in range(NUM_STATES):
        for s2 in range(NUM_STATES+1):
            for s3 in range(NUM_STATES+1):
                s = bytes((s1, s2, s3))
                transitions[s] = {}
                for a in range(NUM_ACTIONS):
                    transitions[s][a] = []
                    index_entry = index_cached[index_offset:index_offset+BYTES_INDEX]
                    if index_entry == b'':
                        break
                    file_id = bytes2long(index_entry, 0, 1)
                    file_offset = bytes2long(index_entry, 1, BYTES_INDEX-1)
                    row = files_cached[file_id][file_offset:file_offset+(BYTES_FIELD*file_id)]
                    for i in range(file_id):
                        inp = row[i*BYTES_FIELD:(i+1)*BYTES_FIELD]
                        if inp == b'':
                            break
                        transitions[s][a].append((1.0/file_id, inp[0:3], inp[3] if inp[3] < 128 else inp[3] % -(1 << 8), bool(inp[4])))

                    index_offset += BYTES_INDEX

    print("done")
    return transitions

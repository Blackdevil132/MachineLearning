from defines import *
import numpy as np


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
        if step[2][1] == 255:
            output.append("SLAYING Enemy at %i." % step[0][1])
        elif step[2][2] == 255:
            output.append("SLAYING Enemy at %i." % step[0][2])
        else:
            output.append("SLAYED Nothing.")
    else:
        output.append("Moving from %i %s to %i." % (step[0][0], IntToAction[step[1]], step[2][0]))

    if step[2][1] == 255:
        output.append("Enemy 1 is DEAD.")
    elif step[0][1] == step[2][1]:
        output.append("Enemy 1 is STAYING at %i." % step[2][1])
    else:
        output.append("Enemy 1 moving from %i to %i." % (step[0][1], step[2][1]))

    if step[2][2] == 255:
        output.append("Enemy 2 is DEAD.")
    elif step[0][2] == step[2][2]:
        output.append("Enemy 2 is STAYING at %i." % step[2][2])
    else:
        output.append("Enemy 2 moving from %i to %i." % (step[0][2], step[2][2]))
    output.append("Reward: %i." % step[3])
    return "{: <30} {: <30} {: <30} {: <20}".format(*output)


def getEnemyPattern(nrow, ncol, nA_e):
    pattern = {i: np.zeros(nA_e) for i in range(nrow * ncol)}
    for row in range(nrow):
        for col in range(ncol):
            possible_moves = np.zeros(nA_e)
            for a in range(nA_e):
                newrow, newcol = inc(row, col, a, nrow, ncol)
                if a != STAY and to_s(row, col, ncol) == to_s(newrow, newcol, ncol):
                    pattern[to_s(row, col, ncol)][a] = 0
                else:
                    possible_moves[a] = True

            for a in range(nA_e):
                if possible_moves[a]:
                    pattern[to_s(row, col, ncol)][a] = 1.0 / possible_moves.sum()

    pattern[255] = np.zeros(nA_e)
    pattern[255][0] = 1.0
    return pattern


def to_s(row, col, ncol):
    return row * ncol + col


def inc(row, col, a, nrow, ncol):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    elif a in [STAY, SLAY]:
        pass
    return (row, col)
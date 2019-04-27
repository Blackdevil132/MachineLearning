from src.GameEnemy import GameEnemy
from src.QRL import QRL
from src.Game2Enemies import Game2Enemies
import numpy as np
from defines import *

#qrl = QRL(Game2Enemies(map_name="8x8"))
#qrl.loadFromFile("qtables/190426_11")
#qrl.test_visual()
nA_e = 6
nrow, ncol = 8, 8

def to_s(row, col):
    return row * ncol + col


def inc(row, col, a):
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

def getEnemyPattern(nrow, ncol):
    pattern = {i: np.zeros(nA_e) for i in range(nrow * ncol)}
    for row in range(nrow):
        for col in range(ncol):
            possible_moves = np.zeros(nA_e)
            for a in range(nA_e):
                newrow, newcol = inc(row, col, a)
                if a not in [STAY, SLAY] and to_s(row, col) == to_s(newrow, newcol):
                    pattern[to_s(row, col)][a] = 0
                else:
                    possible_moves[a] = True

            for a in range(nA_e):
                if possible_moves[a]:
                    pattern[to_s(row, col)][a] = 1.0 / possible_moves.sum()

    pattern[64] = np.zeros(nA_e)
    pattern[64][0] = 1.0
    return pattern


enemy_pattern = getEnemyPattern(nrow, ncol)


def getEnemyMoves(s):
    moves = []
    row_e, col_e = s // ncol, s % ncol
    for a_e in range(nA_e):
        prob_a_e = enemy_pattern[s][a_e]
        newrow_e, newcol_e = inc(row_e, col_e, a_e)
        new_s_e = to_s(newrow_e, newcol_e)
        moves.append((prob_a_e, new_s_e, a_e))

    return moves


print(getEnemyPattern(8, 8))

"""
done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()"""

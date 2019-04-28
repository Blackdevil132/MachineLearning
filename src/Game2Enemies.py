import sys, pickle
import numpy as np
from six import StringIO
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete
from defines import *
from src.tools.helpers import loadTransitions, saveTransitions


class Game2Enemies(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4"):
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = {b'F': 0, b'H': -100, b'G': 100, b'S': 0, b'K': 50, b'P': -100}

        nA = 6
        nA_e = 6
        nS = (self.nrow * self.ncol) * ((self.nrow * self.ncol) + 1) * ((self.nrow * self.ncol) + 1)

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        try:
            P = loadTransitions()
        except FileNotFoundError:
            print("Computing Transition Matrix for Game...")
            P = {bytes((s, e1, e2)): {a: [] for a in range(nA)} for s in range(self.ncol*self.nrow) for e1 in range(self.ncol*self.nrow + 1) for e2 in range(self.ncol*self.nrow + 1)}

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
                pattern = {i: np.zeros(nA_e) for i in range(nrow*ncol)}
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
                row_e, col_e = s // self.ncol, s % self.ncol
                for a_e in range(nA_e):
                    prob_a_e = enemy_pattern[s][a_e]
                    newrow_e, newcol_e = inc(row_e, col_e, a_e)
                    new_s_e = to_s(newrow_e, newcol_e)
                    moves.append((prob_a_e, new_s_e, a_e))

                return moves

            for row in range(nrow):
                for col in range(ncol):
                    for s_e in range(nrow*ncol + 1):
                        row_e, col_e = s_e // ncol, s_e % ncol
                        for s_e2 in range(nrow*ncol + 1):
                            row_e2, col_e2 = s_e2 // ncol, s_e2 % ncol
                            s = bytes((to_s(row, col), s_e, s_e2))

                            for a in range(nA):
                                li = P[s][a]
                                letter = desc[row, col]
                                if letter in b'GH':
                                    # Finite State
                                    li.append((1.0, s, self.reward_range[letter], True))
                                else:
                                    # non finite state
                                    newrow, newcol = inc(row, col, a)
                                    new_s = to_s(newrow, newcol)
                                    if new_s in [s[1], s[2]]:
                                        # Agent moved into Enemy; not allowed
                                        rew = self.reward_range[b'P']
                                        li.append((1.0, bytes((new_s, s[1], s[2])), rew, True))
                                    else:
                                        # allowed movement
                                        for move_e in getEnemyMoves(s[1]):
                                            for move_e2 in getEnemyMoves(s[2]):
                                                if move_e[0] * move_e2[0] != 0:
                                                    new_s_e, new_s_e2 = move_e[1], move_e2[1]

                                                    if a == SLAY:
                                                        # slaying
                                                        adjacent = [to_s(*inc(row, col, d)) for d in range(1, 5)]
                                                        done = False
                                                        rew = self.reward_range[b'P']
                                                        if s[1] in adjacent and s[0] != s[1]:
                                                            if move_e[2] == SLAY:
                                                                rew = 0
                                                                new_s_e = s_e
                                                            else:
                                                                rew = self.reward_range[b'K']
                                                                new_s_e = 64
                                                        if s[2] in adjacent and s[0] != s[2]:
                                                            # if slaying and enemy 2 is adjacent
                                                            if move_e2[2] == SLAY:
                                                                # saved, if enemy is slaying aswell
                                                                rew = 0
                                                                new_s_e2 = s_e2
                                                            else:
                                                                # dead if not
                                                                rew = self.reward_range[b'K']
                                                                new_s_e2 = 64
                                                    else:
                                                        newletter = desc[newrow, newcol]
                                                        # moving or staying
                                                        if move_e[2] == SLAY and new_s in [to_s(*inc(row_e, col_e, d)) for d in range(1, 5)]:
                                                            # moving next to enemy 1 who is attacking, dead
                                                            done = True
                                                        elif move_e2[2] == SLAY and new_s in [to_s(*inc(row_e2, col_e2, d)) for d in range(1, 5)]:
                                                            # moving next to enemy 2 who is attacking, dead
                                                            done = True
                                                        else:
                                                            done = bytes(newletter) in b'GH' or new_s in [new_s_e, new_s_e2]

                                                        # penalize moving out of bounds
                                                        if a != STAY and new_s == s[0]:
                                                            rew = self.reward_range[b'P']
                                                        else:
                                                            rew = self.reward_range[newletter]

                                                    newstate = bytes((new_s, new_s_e, new_s_e2))
                                                    li.append((move_e[0] * move_e2[0], newstate, rew, done))

            saveTransitions(P)

        super(Game2Enemies, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s[0] // self.ncol, self.s[0] % self.ncol
        row_e, col_e = self.s[1] // self.ncol, self.s[1] % self.ncol
        row_e2, col_e2 = self.s[2] // self.ncol, self.s[2] % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "blue", highlight=True)
        try:
            desc[row_e][col_e] = utils.colorize(desc[row_e][col_e], "red", highlight=True)
        except IndexError:
            pass
        try:
            desc[row_e2][col_e2] = utils.colorize(desc[row_e2][col_e2], "red", highlight=True)
        except IndexError:
            pass
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Stay", "Left", "Down", "Right", "Up", "Slay"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        self.s = bytes((self.ncol-1, 0, (self.nrow-1)*self.ncol))
        self.lastaction = None
        return self.s

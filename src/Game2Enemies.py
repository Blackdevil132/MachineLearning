import sys, pickle
import numpy as np
from six import StringIO
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete
from defines import *
from src.tools.helpers import getEnemyPattern


class Game2Enemies(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4"):
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = {b'F': 0, b'H': -100, b'G': 200, b'S': 0, b'K': 100, b'P': -100}

        nA = 6
        self.nA_e = 5
        self.enemy_pattern = getEnemyPattern(self.nrow, self.ncol, self.nA_e)
        nS = (self.nrow * self.ncol)**3

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = None

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
        self.s = bytes((0, self.ncol-1, (self.nrow-1)*self.ncol))
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = []

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            elif a in [STAY, SLAY]:
                pass
            return (row, col)

        def getEnemyMoves(s):
            moves = []
            row_e, col_e = s // self.ncol, s % self.ncol
            for a_e in range(self.nA_e):
                prob_a_e = self.enemy_pattern[s][a_e]
                newrow_e, newcol_e = inc(row_e, col_e, a_e)
                new_s_e = to_s(newrow_e, newcol_e)
                moves.append((prob_a_e, new_s_e))

            return moves

        row, col = self.s[0] // self.ncol, self.s[0] % self.ncol
        letter = self.desc[row, col]
        if letter in b'GH':
            # Finite State
            transitions.append((1.0, self.s, self.reward_range[letter], True))
        else:
            # non finite state
            newrow, newcol = inc(row, col, a)
            new_s = to_s(newrow, newcol)
            if new_s in [self.s[1], self.s[2]]:
                # Agent moved into Enemy; not allowed
                rew = self.reward_range[b'P']
                transitions.append((1.0, bytes((new_s, self.s[1], self.s[2])), rew, True))
            else:
                adjacent = [to_s(*inc(row, col, d)) for d in range(4)]
                # if enemy 1 is adjacent
                if a == SLAY and self.s[1] in adjacent and self.s[0] != self.s[1]:
                    for move in getEnemyMoves(self.s[2]):
                        newstate = bytes((new_s, 255, move[1]))
                        rew = self.reward_range[b'K']
                        transitions.append((move[0], newstate, rew, False))

                # if enemy 2 is adjacent
                elif a == SLAY and self.s[2] in adjacent and self.s[0] != self.s[2]:
                    for move in getEnemyMoves(self.s[1]):
                        newstate = bytes((new_s, 255, move[1]))
                        rew = self.reward_range[b'K']
                        transitions.append((move[0], newstate, rew, False))

                # normal movement
                else:
                    for move_e1 in getEnemyMoves(self.s[1]):
                        for move_e2 in getEnemyMoves(self.s[2]):
                            # enemy 1 movement

                            newstate = bytes((new_s, move_e1[1], move_e2[1]))
                            newletter = self.desc[newrow, newcol]
                            done = bytes(newletter) in b'GH' or newstate[0] in [newstate[1], newstate[2]]
                            # penalize moving out of bounds
                            if a != STAY and newstate[0] == self.s[0]:
                                rew = self.reward_range[b'P']
                            else:
                                rew = self.reward_range[newletter]
                            transitions.append((move_e1[0] * move_e2[0], newstate, rew, done))

        for t in transitions:
            print(t)
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob": p})

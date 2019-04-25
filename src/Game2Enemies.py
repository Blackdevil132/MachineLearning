import sys, pickle
import numpy as np
from six import StringIO
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete
from defines import *


class Game2Enemies(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4"):
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = {b'F': 0, b'H': -100, b'G': 200, b'S': 0, b'K': 100, b'P': -100}

        nA = 6
        nA_e = 5
        nS = (self.nrow * self.ncol)**3

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        try:
            with open("transitions.pkl", 'rb') as file:
                P = pickle.load(file)
                print("Loaded Transition Matrix from file.")
        except FileNotFoundError:
            print("Computing Transition Matrix for Game...")
            P = {bytes((s, e1, e2)): {a: [] for a in range(nA)} for s in range(self.ncol*self.nrow) for e1 in range(self.ncol*self.nrow) for e2 in range(self.ncol*self.nrow)}

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
                            if a != STAY and to_s(row, col) == to_s(newrow, newcol):
                                pattern[to_s(row, col)][a] = 0
                            else:
                                possible_moves[a] = True

                        for a in range(nA_e):
                            if possible_moves[a]:
                                pattern[to_s(row, col)][a] = 1.0 / possible_moves.sum()

                pattern[255] = np.zeros(nA_e)
                pattern[255][0] = 1.0
                return pattern

            enemy_pattern = getEnemyPattern(nrow, ncol)

            for row in range(nrow):
                for col in range(ncol):
                    for s_e in [i for i in range(nrow*ncol)] + [255]:
                        row_e, col_e = s_e // ncol, s_e % ncol
                        for s_e2 in [j for j in range(nrow*ncol)] + [255]:
                            row_e2, col_e2 = s_e2 // ncol, s_e2 % ncol
                            s = bytes((to_s(row, col), s_e, s_e2))
                            # enemy 1 is dead
                            if s_e == 255 or s_e2 == 255:
                                P[s] = {a: [] for a in range(nA)}

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
                                        newletter = desc[newrow, newcol]
                                        rew = self.reward_range[b'P']
                                        done = bytes(newletter) in b'GH'
                                        # TODO maybe fix wrong state assignment
                                        li.append((0.0, bytes((new_s, s[1], s[2])), rew, done))
                                    else:
                                        adjacent = [to_s(*inc(row, col, d)) for d in range(4)]
                                        # if enemy 1 is adjacent
                                        if a == SLAY and s[1] in adjacent and s[0] != s[1]:
                                            for a_e2 in range(nA_e):
                                                prob_a_e2 = enemy_pattern[to_s(row_e2, col_e2)][a_e2]
                                                newrow_e2, newcol_e2 = inc(row_e2, col_e2, a_e2)
                                                new_s_e2 = to_s(newrow_e2, newcol_e2)
                                                newstate = bytes((new_s, 255, new_s_e2))
                                                rew = self.reward_range[b'K']
                                                li.append((prob_a_e2, newstate, rew, False))

                                        # if enemy 2 is adjacent
                                        elif a == SLAY and s[2] in adjacent and s[0] != s[2]:
                                            for a_e in range(nA_e):
                                                prob_a_e = enemy_pattern[to_s(row_e, col_e)][a_e]
                                                newrow_e, newcol_e = inc(row_e, col_e, a_e)
                                                new_s_e = to_s(newrow_e, newcol_e)
                                                newstate = bytes((new_s, new_s_e, 255))
                                                rew = self.reward_range[b'K']
                                                li.append((prob_a_e, newstate, rew, False))

                                        # normal movement
                                        else:
                                            for a_e in range(nA_e if s_e != 255 else 1):
                                                for a_e2 in range(nA_e if s_e2 != 255 else 1):
                                                    # enemy 1 movement
                                                    prob_a_e = enemy_pattern[s_e][a_e]
                                                    newrow_e, newcol_e = inc(row_e, col_e, a_e)
                                                    new_s_e = to_s(newrow_e, newcol_e)

                                                    # enemy 2 movement
                                                    prob_a_e2 = enemy_pattern[s_e2][a_e2]
                                                    newrow_e2, newcol_e2 = inc(row_e2, col_e2, a_e2)
                                                    new_s_e2 = to_s(newrow_e2, newcol_e2)
                                                    newstate = bytes((new_s, new_s_e, new_s_e2))
                                                    newletter = desc[newrow, newcol]
                                                    done = bytes(newletter) in b'GH' or newstate[0] in [newstate[1], newstate[2]]
                                                    # penalize moving out of bounds
                                                    if a != STAY and newstate[0] == s[0]:
                                                        rew = self.reward_range[b'P']
                                                    else:
                                                        rew = self.reward_range[newletter]
                                                    li.append((prob_a_e*prob_a_e2, newstate, rew, done))

            with open("transitions.pkl", 'wb') as file:
                pickle.dump(P, file, pickle.HIGHEST_PROTOCOL)

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
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
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

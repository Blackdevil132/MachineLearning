import sys
import numpy as np
from six import StringIO
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


class GameEnemy(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=False):
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = {b'F': 0, b'H': -10, b'G': 1000, b'S': 0}

        nA = 4
        nS = (self.nrow * self.ncol)**2

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {bytes((s, i)): {a: [] for a in range(nA)} for s in range(self.ncol*self.nrow) for i in range(self.ncol*self.nrow)}

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
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                for row_e in range(nrow):
                    for col_e in range(ncol):
                        s = bytes((to_s(row, col), to_s(row_e, col_e)))
                        for a in range(4):
                            li = P[s][a]
                            letter = desc[row, col]
                            if letter in b'GH':
                                li.append((1.0, s, self.reward_range[letter], True))
                            else:
                                newrow, newcol = inc(row, col, a)
                                for a_e in range(4):
                                    newrow_e, newcol_e = inc(row_e, col_e, a_e)
                                    newstate = bytes((to_s(newrow, newcol), to_s(newrow_e, newcol_e)))
                                    newletter = desc[newrow, newcol]
                                    done = bytes(newletter) in b'GH' or newstate[0] == newstate[1]
                                    # penalize moving out of bounds
                                    if newstate[0] == s[0]:
                                        rew = -10
                                    else:
                                        rew = self.reward_range[newletter]
                                    li.append((1.0 / 4.0, newstate, rew, done))

        super(GameEnemy, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s[0] // self.ncol, self.s[0] % self.ncol
        row_e, col_e = self.s[1] // self.ncol, self.s[1] % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        desc[row_e][col_e] = utils.colorize(desc[row_e][col_e], "blue", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        self.s = bytes((0, self.ncol-1))
        self.lastaction = None
        return self.s

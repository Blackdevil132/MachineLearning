import sys
import numpy as np
from six import StringIO
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete
from src.tools.helpers import generate_random_map


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


class Game(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=False, max_steps=12):
        if desc is None and map_name is None:
            desc = generate_random_map(4, 0.5)
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = {b'F': -1, b'H': -10, b'G': 1000, b'S': -10, b'T': 5}
        self.max_steps = max_steps

        self.desc[3][6] = b'T'

        nA = 4
        nS = self.nrow * self.ncol * self.max_steps

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {bytes((s, i)): {a: [] for a in range(nA)} for s in range(self.ncol*self.nrow) for i in range(self.max_steps)}

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
                for step in range(max_steps):
                    s = bytes((to_s(row, col), step))
                    for a in range(4):
                        li = P[s][a]
                        letter = desc[row, col]
                        if letter in b'GH':
                            li.append((1.0, s, self.reward_range[letter], True))
                        else:
                            if is_slippery:
                                for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                    newrow, newcol = inc(row, col, b)
                                    newstate = bytes((to_s(newrow, newcol), step+1))
                                    newletter = desc[newrow, newcol]
                                    done = bytes(newletter) in b'GH' or step == self.max_steps
                                    rew = self.reward_range[newletter]
                                    li.append((1.0 / 3.0, newstate, rew, done))
                            else:
                                newrow, newcol = inc(row, col, a)
                                newstate = bytes((to_s(newrow, newcol), step+1))
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH' or step == self.max_steps
                                if newstate[0] == s[0]:
                                    rew = -100
                                else:
                                    rew = self.reward_range[newletter]
                                li.append((1.0, newstate, rew, done))

        super(Game, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s[0] // self.ncol, self.s[0] % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        self.s = bytes((0, 0))
        self.lastaction = None
        return self.s

import numpy as np
import pickle


class Qtable:
    def __init__(self, action_space, observation_space, max_steps):
        self.action_space = action_space
        self.observation_space = observation_space
        self.max_steps = max_steps

        self.table = [[np.zeros(action_space) for j in range(max_steps)] for i in range(observation_space)]

    def get(self, state, action=None):
        if action is None:
            return self.table[state[0]][state[1]][:]

        return self.table[state[0]][state[1]][action]

    def update(self, state, action, newValue):
        self.table[state[0]][state[1]][action] = newValue

    def show(self):
        for state in range(self.observation_space):
            print("%i " % state, end='')
            for step in range(self.max_steps):
                print("\t%i: " % step, end='')
                for action in self.table[state][step]:
                    print("\t%.3f, " % action, end='')
                print()

    def fromFile(self, path):
        with open(path + '.pkl', 'rb') as f:
            self.table = pickle.load(f)

    def toFile(self, path):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self.table, f, pickle.HIGHEST_PROTOCOL)

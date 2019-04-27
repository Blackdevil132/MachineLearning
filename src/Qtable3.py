import numpy as np
from src.Qtable import Qtable


# Qtable for 3-dim storing
class Qtable3(Qtable):
    def __init__(self, action_space, observation_space_1, observation_space_2, observation_space_3):
        Qtable.__init__(self)
        self.action_space = action_space
        self.observation_space = (observation_space_1, observation_space_2, observation_space_3)

        self.table = [{j: {k: np.zeros(action_space) for k in [e2 for e2 in range(observation_space_3)]} for j in [e1 for e1 in range(observation_space_2)]} for i in range(observation_space_1)]

    def get(self, state, action=None):
        #print(state[0], state[1], state[2])
        if action is None:
            try:
                return self.table[state[0]][state[1]][state[2]][:]
            except KeyError:
                print("Error at ", state[0], state[1], state[2])

        return self.table[state[0]][state[1]][state[2]][action]

    def update(self, state, action, newValue):
        self.table[state[0]][state[1]][state[2]][action] = newValue

    def show(self):
        for dim1 in range(self.observation_space[0]):
            print("%i " % dim1, end='')
            for dim2 in self.table[dim1].keys():
                print("%i " % dim2, end='')
                for key in self.table[dim1][dim2].keys():
                    print("\t%i: " % key, end='')
                    for action in self.table[dim1][dim2][key]:
                        print("\t%.3f, " % action, end='')
                    print()

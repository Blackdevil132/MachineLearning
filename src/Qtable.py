import pickle


class Qtable:
    """
    base class for qtables
    contains save/store functions
    get/update have to be overridden if qtable has multi dimensional format
    """
    def __init__(self):
        self.table = {}

    def fromFile(self, path):
        """
        Loads QTable from given Path
        :param path: path to load from
        """
        with open(path + '.pkl', 'rb') as f:
            self.table = pickle.load(f)

    def toFile(self, path):
        """
        Saves Qtable in .pkl file
        :param path: Path to save file at
        """
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self.table, f, pickle.HIGHEST_PROTOCOL)

    def get(self, state, action=None):
        """
        :param state: state of environment
        :param action: one possible action for given state
        :return: QTable Value for state and action
        """
        if action is None:
            return self.table[state][:]

        return self.table[state][action]

    def update(self, state, action, newValue):
        """
        :param state: state of environment
        :param action: possible action for given state
        :param newValue: Value to assign
        """
        self.table[state][action] = newValue

    def show(self):
        """
        pretty prints the qtable
        """
        for state in self.table.keys():
            print("%i " % state, end='')
            for action in self.table[state]:
                print("\t%.3f, " % action, end='')
            print()

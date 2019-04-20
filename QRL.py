import sys
import numpy as np
import random

import Game
import pickle


# total_episodes = 500000        # Total episodes
# learning_rate = 0.8           # Learning rate
max_steps = 99  # Max steps per episode
# gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability


# decay_rate = 0.004             # Exponential decay rate for exploration prob


class QRL:
    def __init__(self, total_episodes, learning_rate, discount_rate, decay_rate):
        self.total_episodes = total_episodes
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.decay_rate = decay_rate

        #self.action_space = space[0]
        #self.observation_space = space[1]

        self.qtable = {}
        self.environment = Game.Game(map_name="8x8", is_slippery=False)

        print("Initialized QRL with Parameters: %i, %i, %i, %i" % (total_episodes, learning_rate, discount_rate, decay_rate))

    def statusBar(self, iteration):
        bar_len = 60
        filled_len = int(round(bar_len * iteration / self.total_episodes))
        percents = 100 * iteration / self.total_episodes
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('\r[%s] %s%%\n' % (bar, percents))
        sys.stdout.flush()

    def exportToFile(self, path="qtable"):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self.qtable, f, pickle.HIGHEST_PROTOCOL)

    def loadFromFile(self, path="qtable"):
        with open(path + '.pkl', 'rb') as f:
            self.qtable = pickle.load(f)

    def updateQ(self, state, action, new_state, reward, done):

        if new_state not in self.qtable:
            self.qtable[new_state] = np.zeros(self.environment.action_space.n)

        #print("Updating Q-Value for %i, %i" % (state, action))
        # if state is unknown, add empty entry to qtable
        if state not in self.qtable:
            #print("Adding new QTable Entry..")
            self.qtable[state] = np.zeros(self.environment.action_space.n)

        if done:
            self.qtable[state][action] = reward

        #print("Old QValue: %.2f" % self.qtable[state][action])
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        self.qtable[state][action] = self.qtable[state][action] + self.learning_rate * (
                reward + self.discount_rate * np.max(self.qtable[new_state][:]) - self.qtable[state][action])

        #print("New QValue: %.2f" % self.qtable[state][action])

    def updateEpsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def getMaxFutureReward(self, next_state):
        possible_future_rewards = [0]
        # check all possible next states and get max rewards for all possible enemy actions
        for a in self.environment.action_space:
            try:
                future_reward = self.qtable[next_state][a]
            except KeyError:
                future_reward = 0
            possible_future_rewards.append(future_reward)

        return np.max(possible_future_rewards)

    def learnFromSteps(self, list_of_steps):
        # iterate through all steps taken by the agent from last to first and learn
        for step in list_of_steps:
            # update qtable-entry for current state and action
            self.updateQ(*step)

    def run(self):
        # execute Game and learn
        for episode in range(self.total_episodes):
            # display progress bar
            if episode % (self.total_episodes/100) == 0:
                self.statusBar(episode)

            # Reset the environment
            state = self.environment.reset()

            # execute one iteration of the game
            steps = []
            done = False
            while not done and len(steps) < max_steps:
                action = self.getNextAction(state)
                new_state, reward, done, p = self.environment.step(action)
                steps.append((state, action, new_state, reward, done))
                state = new_state

            # iterate through all steps taken by the agent from last to first and learn
            self.learnFromSteps(steps)

            # Reduce epsilon
            self.updateEpsilon(episode)

        self.exportToFile()

    def getNextAction(self, state, test=False):
        exp_exp_tradeoff = random.uniform(0, 1)

        if test:
            exp_exp_tradeoff = self.epsilon+1

        if exp_exp_tradeoff > self.epsilon:
            # exploit
            try:
                actions = np.where(self.qtable[state][:] == np.max(self.qtable[state][:]))[0]
                action = random.choice(actions)
            # take random action, if qtable has no values yet
            except KeyError:
                action = self.environment.action_space.sample()
        else:
            # explore
            action = self.environment.action_space.sample()

        return action

    def test(self, render=True):
        # Reset the environment
        # execute one iteration of the game
        steps = []
        done = False
        state = self.environment.reset()
        while not done:
            if render:
                self.environment.render("human")
            action = self.getNextAction(state, True)
            new_state, reward, done, p = self.environment.step(action)
            steps.append((state, action, new_state, reward))
            state = new_state

        return steps

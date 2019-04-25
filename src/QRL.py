import numpy as np
import random
import datetime
import time
import sys
import pygame as pg

from src.pgassets.common.pgGrid import pgGrid
from src.pgassets.game.pgField import pgField
from src.pgassets.common.pgTextPanel import pgTextPanel

from src.Game2Enemies import Game2Enemies
from src.Qtable3 import Qtable3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR_BG = (230, 230, 230)
PATH = "qtables/"
MAP = bytes("FFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG", "utf8")
max_steps = 40  # Max steps per episode


class QRL:
    def __init__(self, env=None, learning_rate=0.8, discount_rate=0.9, decay_rate=0.0001):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.decay_rate = decay_rate

        self.qtable = Qtable3(6, 64, 64, 64)
        self.environment = Game2Enemies(map_name="4x4") if env is None else env

        self.exportPath = None

        self.expexpratio = [0, 0]

    def exportToFile(self):
        if self.exportPath is None:
            date = datetime.datetime.today().strftime("%y%m%d_%H")
            path = PATH + date
            self.exportPath = path
        print("Storing Q-Table in %s.pkl... " % self.exportPath)
        self.qtable.toFile(self.exportPath)

    def loadFromFile(self, path=None):
        if path is None:
            path = self.exportPath
        self.qtable.fromFile(path)
        print("Loaded Q-Table from %s.pkl" % path)

    def updateQ(self, state, action, new_state, reward, done):
        if done:
            self.qtable.update(state, action, reward)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        oldValue = self.qtable.get(state, action)
        newValue = oldValue + self.learning_rate * (reward + self.discount_rate * np.max(self.qtable.get(new_state)) - oldValue)
        self.qtable.update(state, action, newValue)

    def updateEpsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def learnFromSteps(self, list_of_steps):
        # iterate through all steps taken by the agent from last to first and learn
        for step in list_of_steps:
            # update qtable-entry for current state and action
            self.updateQ(*step)

    def run(self, total_episodes, path=None):
        # Try loading file from path and continue learning. If not exists, save newly learned table in path
        if path is not None:
            self.exportPath = path
            try:
                self.loadFromFile()
            except FileNotFoundError:
                pass

        print("Initialized Learning with Parameters: %i, %.2f, %.2f, %.4f..." % (total_episodes, self.learning_rate, self.discount_rate, self.decay_rate))
        print("Progress: [", end='')
        # execute Game and learn
        for episode in range(total_episodes):
            # display progress bar
            if episode % int(total_episodes/55.0) == 0:
                #statusBar(episode, total_episodes)
                print("=", end='')

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

        print("]")
        self.exportToFile()

    def getNextAction(self, state, test=False):
        if test:
            exp_exp_tradeoff = self.epsilon+1
        else:
            exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > self.epsilon:
            self.expexpratio[0] += 1
            # exploit
            #if np.max(self.qtable.get(state)) == 0:
            #    actions = []
            #    for step in range(max_steps):
            #        actions.append(np.argmax(self.qtable.get(bytes((state[0], step)))))
            #    unique, counts = np.unique(actions, return_counts=True)

            #    action = np.argmax(counts)
            #else:
            actions = np.where(self.qtable.get(state) == np.max(self.qtable.get(state)))[0]
            action = random.choice(actions)
        else:
            self.expexpratio[1] += 1
            # explore
            action = self.environment.action_space.sample()

        return action

    def test(self, render=True):
        # Reset the environment
        # execute one iteration of the game
        steps = []
        done = False
        state = self.environment.reset()
        while not done and max_steps > len(steps):
            if render:
                self.environment.render("human")
            action = self.getNextAction(state, True)
            new_state, reward, done, p = self.environment.step(action)
            steps.append((state, action, new_state, reward))
            state = new_state
        if render:
            self.environment.render()

        return steps

    def test_visual_human(self):
        # init pygame window
        pg.init()
        size = width, height = 1288, 1024
        icon = pg.image.load("images/circuit_icon_32.png")
        pg.display.set_icon(icon)
        screen = pg.display.set_mode(size)
        pg.display.set_caption("QRL Game")

        def update():
            screen.fill(COLOR_BG)
            for asset in assets:
                asset.draw(screen)

            pg.display.flip()

        def getField(state, unit=0):
            units = [b'', b'K', b'E']
            translator = {83: b'F', 70: b'F', 71: b'G', 72: b'H'}
            field = translator[MAP[state]] + units[unit]
            return field

        # init assets
        assets = []
        fields = []
        for i in range(len(MAP)):
            fields.append(pgField((0, 0), (10, 10), getField(i), id=i))
        fields[0].set_type(b'FK')
        fields[63].set_type(b'GE')
        court = pgGrid((0, 0), (1024, 1024), (8, 8), fields, borderwidth=5)
        assets.append(court)

        text_steps = pgTextPanel((1032, 64), (256, 64), "Steps: 0")
        text_reward = pgTextPanel((1032, 128), (256, 64), "Reward: 0")
        assets.append(text_steps)
        assets.append(text_reward)

        update()

        # init Game
        counter = 1
        done = False
        steps = []
        total_reward = 0
        state = self.environment.reset()
        self.environment.render()
        while not done and max_steps > len(steps):
            # handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT:
                        action = 0
                    elif event.key == pg.K_DOWN:
                        action = 1
                    elif event.key == pg.K_RIGHT:
                        action = 2
                    elif event.key == pg.K_UP:
                        action = 3
                    elif event.key == pg.K_SPACE:
                        action = 4
                    else:
                        action = -1

                    if action in (0, 1, 2, 3, 4):
                        newstate, reward, done, info = self.environment.step(action)
                        steps.append((state, action, newstate, reward))

                        total_reward += reward

                        # update assets
                        court.objects[state[0]].set_type(getField(state[0]))
                        court.objects[newstate[0]].set_type(getField(newstate[0], 1))
                        if newstate[1] != 255:
                            court.objects[state[1]].set_type(getField(state[1]))
                            court.objects[newstate[1]].set_type(getField(newstate[1], 2))

                        text_steps.set_text("Steps: %i" % len(steps))
                        text_reward.set_text("Reward: %i" % total_reward)

                        update()

                        state = newstate
                        counter += 1

        pg.quit()
        sys.exit()

    def test_visual(self):
        # init pygame window
        pg.init()
        size = width, height = 1332, 1024
        icon = pg.image.load("images/brain_icon_32.png")
        pg.display.set_icon(icon)
        screen = pg.display.set_mode(size)
        pg.display.set_caption("QRL Game")

        def update():
            screen.fill(COLOR_BG)
            for asset in assets:
                asset.draw(screen)

            pg.display.flip()

        def getField(state, unit=0):
            units = [b'', b'K', b'E', b'DE']
            translator = {83: b'F', 70: b'F', 71: b'G', 72: b'H'}
            field = translator[MAP[state]] + units[unit]
            return field

        text_last_reward = pgTextPanel((1032, 64), (150, 64), "Last Reward: 0")
        force_exit = False
        while not force_exit:

            # init assets
            assets = []
            fields = []
            for i in range(len(MAP)):
                fields.append(pgField((0, 0), (10, 10), getField(i), id=i))
            fields[0].set_type(b'FK')
            fields[63].set_type(b'G')
            fields[7].set_type(b'FE')
            fields[56].set_type(b'FE')
            court = pgGrid((0, 0), (1024, 1024), (8, 8), fields, borderwidth=10)
            assets.append(court)

            text_steps = pgTextPanel((1032, 0), (300, 64), "Steps Remaining: %i" % max_steps)
            text_reward = pgTextPanel((1182, 64), (150, 64), "Reward: 0")
            text_decision = pgTextPanel((1032, 128), (300, 32), np.array2string(self.qtable.get(bytes((0, 7, 56))), precision=0), fontsize=18)
            assets.append(text_steps)
            assets.append(text_reward)
            assets.append(text_last_reward)
            assets.append(text_decision)

            update()

            # init Game
            counter = 1
            done = False
            steps = []
            total_reward = 0
            state = self.environment.reset()

            while not done and max_steps > len(steps) and not force_exit:
                time.sleep(0.4)
                # handle events
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        force_exit = True

                action = self.getNextAction(state, True)
                if action in (0, 1, 2, 3, 4, 5):
                    newstate, reward, done, info = self.environment.step(action)
                    steps.append((state, action, newstate, reward))

                    total_reward += reward

                    if action == 5:
                        print(action, state, newstate)
                        try:
                            court.objects[state[1]].set_type(getField(state[1], 3))
                            court.objects[state[2]].set_type(getField(state[2], 3))
                        except IndexError:
                            court.objects[state[2]].set_type(getField(state[2], 3))

                    # update assets
                    court.objects[state[0]].set_type(getField(state[0]))
                    court.objects[newstate[0]].set_type(getField(newstate[0], 1))
                    if state[1] != 255 and newstate[1] != 255:
                        court.objects[state[1]].set_type(getField(state[1]))
                        court.objects[newstate[1]].set_type(getField(newstate[1], 2))

                    if state[2] != 255 and newstate[2] != 255:
                        court.objects[state[2]].set_type(getField(state[2]))
                        court.objects[newstate[2]].set_type(getField(newstate[2], 2))

                    text_steps.set_text("Steps Remaining: %i" % (max_steps-len(steps)))
                    text_reward.set_text("Reward: %i" % total_reward)
                    dec_str = np.array2string(self.qtable.get(state), precision=0)
                    assets.append(pgTextPanel((1032, 128 + counter*32), (300, 32), dec_str, fontsize=18))
                    update()

                    state = newstate
                    counter += 1

            text_last_reward.set_text("Last Reward: %i" % total_reward)

import sys

import numpy as np

from src.qrl import QRL
from src.tools.tools import timeit
from src.environments.Game2Enemies import Game2Enemies
from src.tools.helpers import stepToString
from defines import *


if len(sys.argv) < 5:
    # initialize standard parameters
    total_episodes = NUM_EPISODES
    learning_rate = LEARNING_RATE
    discount_rate = DISCOUNT_RATE
    decay_rate = DECAY_RATE
else:
    # initialize with given parameter-values
    total_episodes = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    discount_rate = float(sys.argv[3])
    decay_rate = float(sys.argv[4])


qrl = QRL.QRL(env=Game2Enemies(map_name=MAP_NAME), learning_rate=learning_rate, discount_rate=discount_rate, decay_rate=decay_rate)
exec_time = timeit(qrl.run, [total_episodes, "qtables/190427_20"], 1)

# store exec_time in file
with open("performance.csv", 'a') as file:
    s = "%i,%.2f,%.2f,%.5f,%f;\n" % (total_episodes, learning_rate, discount_rate, decay_rate, exec_time)
    file.write(s)

if SHOW_QTABLE:
    print("=== Q-Table ===============================")
    qrl.loadFromFile()
    qrl.qtable.show()

if SHOW_MAP:
    print("====== MAP Layout ==================\n")
    qrl.environment.reset()
    qrl.environment.render()
    print("\n")

numSteps = []
total_rewards = np.zeros(NUM_TESTS)
for i in range(NUM_TESTS):
    steps = qrl.test(render=False)
    numSteps.append(len(steps))
    for step in steps:
        total_rewards[i] += step[3]

    if SHOW_ONLY_SUBPAR and total_rewards[i] >= THRESHOLD:
        continue

    if SHOW_TESTS:
        print("===== Test Game %i =====================" % (i+1))
        for j in range(len(steps)):
            output = stepToString(steps[j])
            print(j if j >= 10 else "%i " % j, output)
        print("Total Reward: %i\n" % total_rewards[i])

print("\n Average Number of Steps taken %.2f" % np.mean(numSteps))
print("\nMedian Reward: %.2f; Mean Reward: %.2f" % (np.median(total_rewards), np.mean(total_rewards)))
print("Minimum Reward: %i; Maximum Reward: %i" % (total_rewards.min(), total_rewards.max()))

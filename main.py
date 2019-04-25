from src import QRL
import numpy as np
from src.tools.tools import timeit
import sys
from src.GameEnemy import GameEnemy

IntToAction = ["LEFT", "DOWN", "RIGHT", "UP", "STAYING", "SLAYING"]
mapname="8x8"
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4
SLAY = 5

if len(sys.argv) < 5:
    # initialize standard parameters
    total_episodes = 200
    learning_rate = 0.25
    discount_rate = 0.9
    decay_rate = 0.0001
else:
    # initialize with given parameter-values
    total_episodes = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    discount_rate = float(sys.argv[3])
    decay_rate = float(sys.argv[4])


qrl = QRL.QRL(env=GameEnemy(map_name=mapname), learning_rate=learning_rate, discount_rate=discount_rate, decay_rate=decay_rate)
exec_time = timeit(qrl.run, [total_episodes, "qtables/190425_11"], 1)

# store exec_time in file
with open("performance.csv", 'a') as file:
    s = "%i,%.2f,%.2f,%.5f,%f;\n" % (total_episodes, learning_rate, discount_rate, decay_rate, exec_time)
    file.write(s)

print("=== Q-Table ===============================")
qrl.loadFromFile()
qrl.qtable.show()

print("====== MAP Layout ==================\n")
qrl.environment.reset()
qrl.environment.render("human")
print("\n")

numTests = 100
total_rewards = np.zeros(numTests)
for i in range(numTests):
    steps = qrl.test(render=False)
    for step in steps:
        total_rewards[i] += step[3]

    if total_rewards[i] == 300:
        continue

    print("===== Test Game %i =====================" % (i+1))
    for j in range(len(steps)):
        step = steps[j]
        step_str = "%i " % j
        if j < 10:
            step_str += " "
        if step[1] == STAY:
            step_str += "STAYING at %i. \t\t\t\t" % step[2][0]
        elif step[1] == SLAY:
            step_str += "SLAYING Enemy at %i. \t\t" % step[0][1]
        else:
            step_str += "Moving from %i %s to %i. \t" % (step[0][0], IntToAction[step[1]], step[2][0])
            if len(step_str) <= 27:
                step_str += "\t\t"

        if step[2][1] == 255:
            step_str += "Enemy DEAD. \t\t\t\t"
        elif step[0][1] == step[2][1]:
            step_str += "Enemy STAYING at %i. \t\t" % step[2][1]
        else:
            step_str += "Enemy moving from %i to %i. " % (step[0][1], step[2][1])
        step_str += "\tReward: %i." % step[3]
        print(step_str)
    print("Total Reward: %i\n" % total_rewards[i])

print("\nMedian Reward: %.2f; Mean Reward: %.2f" % (np.median(total_rewards), np.mean(total_rewards)))
print("Minimum Reward: %i; Maximum Reward: %i" % (total_rewards.min(), total_rewards.max()))
#print("Exploitation-Exploration Ratio: %i:%i\n" % (qrl.expexpratio[0], qrl.expexpratio[1]))

"""
print("===== Statistics Table ==============================")
for i in range(64):
    for j in range(20):
        print(i, j, qrl.statistics[bytes((i, j))])
"""
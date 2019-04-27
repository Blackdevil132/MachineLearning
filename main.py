from src import QRL
import numpy as np
from src.tools.tools import timeit
import sys
from src.Game2Enemies import Game2Enemies
from src.tools.helpers import stepToString


mapname = "8x8"
show_qtable = False


if len(sys.argv) < 5:
    # initialize standard parameters
    total_episodes = 500000
    learning_rate = 0.2
    discount_rate = 0.9
    decay_rate = 0.00012
else:
    # initialize with given parameter-values
    total_episodes = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    discount_rate = float(sys.argv[3])
    decay_rate = float(sys.argv[4])


qrl = QRL.QRL(env=Game2Enemies(map_name=mapname), learning_rate=learning_rate, discount_rate=discount_rate, decay_rate=decay_rate)
exec_time = timeit(qrl.run, [total_episodes, "qtables/190427_18"], 1)

# store exec_time in file
with open("performance.csv", 'a') as file:
    s = "%i,%.2f,%.2f,%.5f,%f;\n" % (total_episodes, learning_rate, discount_rate, decay_rate, exec_time)
    file.write(s)

if show_qtable:
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

    if total_rewards[i] == 400:
        continue

    print("===== Test Game %i =====================" % (i+1))
    for j in range(len(steps)):
        output = stepToString(steps[j])
        print(j if j >= 10 else "%i " % j, output)
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
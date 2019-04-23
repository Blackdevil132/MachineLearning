from tools import timeit, print_sysinfo
from QRL import QRL
import sys
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import freeze_support


def plot_results(benchmarks, bar_labels, n):
    fig = plt.figure(figsize=(10, 8))

    # plot bars
    y_pos = np.arange(len(benchmarks))
    plt.yticks(y_pos, bar_labels, fontsize=16)
    bars = plt.barh(y_pos, benchmarks,
             align='center', alpha=0.4, color='g')

    # annotation and labels

    for ba,be in zip(bars, benchmarks):
        plt.text(ba.get_width() + 2, ba.get_y() + ba.get_height()/2,
                '{0:.2%}'.format(benchmarks[0]/be),
                ha='center', va='bottom', fontsize=12)

    plt.xlabel('time in seconds for n=%s' % n, fontsize=14)
    plt.ylabel('number of processes', fontsize=14)
    t = plt.title('Serial vs. Multiprocessing Performance', fontsize=18)
    plt.ylim([-1, len(benchmarks)+0.5])
    plt.xlim([0, max(benchmarks)*1.1])
    plt.vlines(benchmarks[0], -1, len(benchmarks)+0.5, linestyles='dashed')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) < 5:
        # initialize standard parameters
        total_episodes = 20000
        learning_rate = 0.8
        discount_rate = 0.95
        decay_rate = 0.0001
    else:
        # initialize with given parameter-values
        total_episodes = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        discount_rate = float(sys.argv[3])
        decay_rate = float(sys.argv[4])

    benchmarks = []

    qrl = QRL(total_episodes, learning_rate, discount_rate, decay_rate)
    #benchmarks.append(timeit(qrl.run, [total_episodes], 1, False))
    benchmarks.append(timeit(qrl.run_parallel, [total_episodes, 2], 1, False))

    print_sysinfo()
    plot_results(benchmarks, ['serial', '2'], total_episodes)

    qrl.qtable.show()




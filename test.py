# instantiate and configure the worker pool
from pathos.multiprocessing import ProcessPool

if __name__ == '__main__':
    pool = ProcessPool(nodes=4)

    # do a blocking map on the chosen function
    print(pool.map(pow, [1,2,3,4], [5,6,7,8]))

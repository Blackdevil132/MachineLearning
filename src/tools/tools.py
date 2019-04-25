import time, math, sys


def statusBar(iteration, total_episodes):
    bar_len = 60
    filled_len = int(round(bar_len * iteration / total_episodes))
    percents = 100 * iteration / total_episodes
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r[%s] %s%%\n' % (bar, percents))
    sys.stdout.flush()


def product(li):
    prod = 1
    for elem in li:
        prod *= elem

    return prod


def sieveOfEratosthenes(n):
    # finds all primes < n

    # list of numbers from 2 to n, excluding even numbers
    primes = [2] + [i for i in range(3, n, 2)]

    # start at index 1, index 0 is already dealt with
    i = 1
    while True:
        try:
            # find position of the square of number at index i
            j = primes.index(primes[i] ** 2)
        except ValueError:
            # if square is > n, done
            return primes

        # remove all multiples of primes[i]
        while j < len(primes):
            if primes[j] % primes[i] == 0:
                primes.pop(j)
            else:
                j += 1

        i += 1


def getPrimes(n):
    # returns primes < n
    # reads primes from file if possible
    with open("primes.txt", 'r') as file:
        primes = file.read().split(',')

    primes = list(map(int, primes))

    if n is None:
        return primes

    elif n > primes[-1]:
        primes = sieveOfAtkin(n)

        file = open("primes.txt", 'w')
        text = ','.join(map(str, primes))
        file.write(text)
        file.close()

        return primes

    else:
        return list(filter(lambda x: x < n, primes))


def readPrimes(PATH, limit=None):
    with open(PATH, 'r') as file:
        diffs = file.read().split(',')

    max_p = int(diffs.pop(0))

    if limit is None:
        limit = max_p

    primes = [2, 3]
    for d in diffs:
        p = primes[-1]+2*int(d)

        if p > limit:
            break

        primes.append(p)

    return primes


def savePrimes(PATH, primes):
    with open(PATH, 'w') as file:
        file.write("%i" % primes[-1])
        for i in range(2, len(primes)):
            file.write(",%i" % ((primes[i] - primes[i-1])//2))


def fac(n):
    # return faculty of n
    # recursive
    if n == 0:
        return 1
    return fac(n-1) * n


def gcD(a, b):
    # return greatest common divisor of a and b
    if a == 0:
        return abs(b)

    if b == 0:
        return abs(a)

    while b != 0:
        tmp = a % b
        a = b
        b = tmp

    return abs(a)


def trialDivision(n, partial=False):
    # returns all prime factors of n
    factors = []

    primes = getPrimes(n)

    if not primes:
        return []

    exp = 2
    if partial:
        exp = 3

    i = 0
    while primes[i]**exp <= n:
        p = primes[i]
        if n % p == 0:
            n //= p
            factors.append(p)
        else:
            i += 1

    if n != 1:
        factors.append(n)

    return factors


class TrialDivision:
    def __init__(self, limit=None, partial=False):
        self.primes = readPrimes("primes.p", limit)
        self.partial = partial

    def exec(self, n):
        factors = []

        if not self.primes:
            return []

        exp = 2
        if self.partial:
            exp = 3

        i = 0
        while self.primes[i] ** exp <= n:
            p = self.primes[i]
            if n % p == 0:
                n //= p
                factors.append(p)
            else:
                i += 1

        if n != 1:
            factors.append(n)

        return factors


def distinct(li):
    # removes all duplicates from list li
    return list(dict.fromkeys(li))


def timeit(func, args, loops=None):
    if loops is None:
        start = time.perf_counter()
        ret = func(*args)
        end = time.perf_counter()

        print("Execution Time for %s%s: %.3f ms" % (func.__name__, str(args), (end-start)*1000))

        return ret

    start = time.perf_counter()
    for i in range(loops):
        func(*args)
    end = time.perf_counter()

    print("%i Loops, Average Execution Time for %s%s: %.3f ms" % (loops, func.__name__, str(args), 1000*(end-start)/loops))
    return (end-start)/loops


def sieveOfAtkin(n):
    primes = [2, 3, 5]
    sieve = [False] * (n+1)

    root_n = int(math.sqrt(n))

    x = 1
    while x*x < n:
        y = 1
        while y*y < n:
            n1 = 4*x**2 + y**2
            n2 = 3*x**2 + y**2
            n3 = 3*x**2 - y**2

            #print(x, y, n)
            #print(n1, n2, n3)

            if n1 <= n and n1 % 60 in (1, 13, 17, 29, 37, 41, 49, 53):
                sieve[n1] ^= True

            if n2 <= n and n2 % 60 in (7, 19, 31, 43):
                sieve[n2] ^= True

            if x > y and n3 <= n and n3 % 60 in (11, 23, 47, 59):
                sieve[n3] ^= True

            y += 1
        x += 1

    for x in range(5, n):
        if sieve[x]:
            primes.append(x)

            if x < root_n:
                for y in range(x**2, n+1, x**2):
                    sieve[y] = False

    return primes



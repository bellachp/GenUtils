# std concurrency options in python

import time
import random

from concurrent.futures import ThreadPoolExecutor, as_completed

# the work to distribute
def hello():
    seconds = random.randint(0, 5)
    print(f'Hi {seconds}s')
    time.sleep(seconds)
    print(f'Bye {seconds}s')
    return seconds

# max concurrency is 2
executor = ThreadPoolExecutor(max_workers=2)

# submit the work
a = executor.submit(hello)
b = executor.submit(hello)

# and here we wait for results
for future in as_completed((a, b)):
    print(future.result())


# Want multiple process instead ? It's the same API:

import time
import random

from concurrent.futures import ProcessPoolExecutor, as_completed

def hello():
    seconds = random.randint(0, 5)
    print(f'Hi {seconds}s')
    time.sleep(seconds)
    print(f'Bye {seconds}s')
    return seconds

# Don't forget this for processes, or you'll get in trouble
if __name__ == "__main__":

    executor = ProcessPoolExecutor(max_workers=2)

    a = executor.submit(hello)
    b = executor.submit(hello)

    for future in as_completed((a, b)):
        print(future.result())

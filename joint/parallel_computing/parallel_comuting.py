import multiprocessing as mp
import numpy as np
from multiprocessing import freeze_support, managers
from multiprocessing.managers import BaseManager
from functools import partial

class MyVar:
    def __init__(self) -> None:
        pass
    some_variable = 3

    def initAnotherVariable(self, A):
        self.anotherVariable = A

var = MyVar()

def mySquare(x, y, var):
    return np.sum(x) * np.ones(len(x)) * var.anotherVariable

def main():
    print("No of processes available: ", mp.cpu_count())
    var.initAnotherVariable(5)

    np.random.seed(1)
    arr_x = np.random.randint(0, 100, (100, 200))
    arr_y = np.arange(len(arr_x))[:, np.newaxis] * np.ones(arr_x.shape)

    pool = mp.Pool(processes=mp.cpu_count())
    result = np.array(
        pool.starmap(partial(mySquare, var=var), zip(arr_x, arr_y))
    )
    pool.close()
    print(result)

if __name__ == '__main__':
    freeze_support()
    main()
    var.anotherVariable = 0
    main()
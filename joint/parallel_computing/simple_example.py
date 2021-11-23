import multiprocessing as mp
from multiprocessing import freeze_support
import numpy as np

def mySquare(x):
    return x*x

def main():
    print("No of processes available: ", mp.cpu_count())

    arr = np.arange(10)

    pool = mp.Pool(processes=mp.cpu_count())
    result = np.array(pool.map(mySquare, arr))
    pool.close()
    
    print(result)

if __name__ == '__main__':
    freeze_support()
    main()
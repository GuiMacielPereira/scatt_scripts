from concurrent.futures import ProcessPoolExecutor
import numpy as np

def mySquare(x):
    return x*x

def main():
    arr = range(10)
    result = list(ProcessPoolExecutor().map(mySquare, arr))
    print(result)

if __name__ == '__main__':
    main()
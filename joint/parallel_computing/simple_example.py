
from concurrent.futures import ProcessPoolExecutor

def mySquare(x):
    return x*x

if __name__ == '__main__':
    arr = range(10)
    result = list(ProcessPoolExecutor().map(mySquare, arr))
    print(result) 
       
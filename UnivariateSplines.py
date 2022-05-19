from typing import Tuple, List
import bisect
import matplotlib.pyplot as plt
import numpy as np
"""
These first couples of lines of code are defined to be in the main of the function of this program.
This serves as a way to perform file opening and reading and I can store those values into an array
or list. This is what were told to do in Question #3 but I'm doing this a little earlier because the
matrix is useful in all function I have created below.
"""
DataPoints = [] #Initialize matrix
with open("Project6_dataset.txt", "r") as file:
    N = file.readlines()
    for lines in N:
        nums = lines.split()
        DataPoints.append(nums)
X, Y = zip(*DataPoints)
X = list(X)
Y = list(Y)
XList = []
YList = []
for element in X:
    XList.append(float(element))
for element in Y:
    YList.append(float(element))
D = np.array(DataPoints)
#These are some helpers methods for my next functions
def changes(x: List[float]) -> List[float]:
    return [x[i+1] - x[i] for i in range(len(x) - 1)]

def tridiagonalmatrix(n: int, h: List[float]) -> Tuple[List[float], List[float], List[float]]:
    A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
    B = [2] * n
    C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
    return A, B, C

def target(n: int, h: List[float], y: List[float]):
    return [0] + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i-1]) for i in range(1, n - 1)] + [0]
"""
This function stores the Xj values in in the Vector X 
"""
def Evaluate_spline(A: List[float], B: List[float], C: List[float], D: List[float]):
    c_p = C + [0]
    d_p = [0] * len(B)
    X = [0] * len(B)

    c_p[0] = C[0] / B[0]
    d_p[0] = D[0] / B[0]
    for i in range(1, len(B)):
        c_p[i] = c_p[i] / (B[i] - c_p[i - 1] * A[i - 1])
        d_p[i] = (D[i] - d_p[i - 1] * A[i - 1]) / (B[i] - c_p[i - 1] * A[i - 1])

    X[-1] = d_p[-1]
    for i in range(len(B) - 2, -1, -1):
        X[i] = d_p[i] - c_p[i] * X[i + 1]

    return X
"""
This is the first function in question #1. Its takes a nx2-dimensional array of the 
n number of datapoints we generated from the .txt file. We break them down to their
respective X and Y column vectors to be evaluated later on.
"""
def Spline(D):
    X, Y = zip(*DataPoints)
    xa = list(X)
    ya = list(Y)
    x = []
    y = []
    for element in xa:
        x.append(float(element))
    for element in ya:
        y.append(float(element))
    
    n = len(x)
    if n != len(y):
        raise ValueError('Array lengths are different')

    h = changes(x)
    if any(v < 0 for v in h):
        raise ValueError('X must be strictly increasing')

    A, B, C = tridiagonalmatrix(n, h)
    D = target(n, h, y)

    M = Evaluate_spline(A, B, C, D)
    #This is the array which will saves the index of the coefficient into an array like its says in
    #Question 1
    coefficients = [[(M[i+1]-M[i])*h[i]*h[i]/6, M[i]*h[i]*h[i]/2, (y[i+1] - y[i] - (M[i+1]+2*M[i])*h[i]*h[i]/6), y[i]] for i in range(n-1)]
    #This nested function serves us too evaluate the coefficients and constants into one function which
    #can be used to graph the function which I will do in the main function later on.
    def spline(val):
        idx = min(bisect.bisect(x, val)-1, n-2)
        z = (val - x[idx]) / h[idx]
        C = coefficients[idx]
        return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]

    return spline

spline = Spline(D)

Xj = [v / 10 for v in range(0, 50, 1)]
Yj = [spline(y) for y in Xj]

#Sets limits for the Y-axis
plt.ylim(-1,1)
#Sets limits for the X-axis
plt.xlim(1,5)
#show the plot and grid points
plt.grid()
#This graphs the points on the datasets from the set and
# the s is the size of the Marker which I set to 8 and the color to red
# which its says to do in 5.1
plt.scatter(XList, YList, s=8, color='red', label="Data-points")
# show the plot of Xg and Yg
plt.xlim(1,5)
plt.ylim(-1,1)
plt.plot(Xj, Yj, color='b', label="Evaluated Spline")
#Lastly we evaluate a function and plot it to the graph that its says to do in
#Question 5
x = np.linspace(1, 5, 100)
y = np.sin(0.1 * x) * np.cos(2 * x) + np.sin(3 * x) * np.sin(0.7 * x)
plt.plot(x, y, color='red', label="sin(0.1x)cos(2x) + sin(3x)sin(0.7x)")
plt.legend()
plt.grid(True, linestyle =':')
plt.show()
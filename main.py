import numpy as np
import matplotlib.pyplot as plt

# This program will illustrate linear regression

# Utility functions
def data_gen(size):
    """This will generate the data for the algorithm"""
    dat = []
    a = 10
    b = 20
    for i in range(size):
        d = np.random.random() * 2 - 1 # random from -1 to 1
        dat.append(a + b * i + round(d * 10 * np.sqrt(i), 2) + round(d * 2 * i, 2) + round(d * 5 / (i+1) ** 3, 2))
    return dat

n = 100 # size of the data

dat = data_gen(n)

def regression(dat):
    A = np.matrix([[0, 0], [0, 0]])
    b = np.matrix([[0], [0]])

    for i in range(len(dat)):
        dA = np.matrix([[1, i], [i, i ** 2]])
        db = np.matrix([[dat[i]], [dat[i] * i]])
        A = A + dA
        b = b + db
    r = A.getI() @ b
    r = np.round(r, decimals=2)
    return r

pred = regression(dat)
a, b = pred.item(0), pred.item(1)

#plot the data
plt.plot(dat, 'ro')
plt.ylabel('y value')
plt.xlabel('x value')

#plot the prediction using lr
x = np.linspace(0, n + 30, 100)
y = a + b * x
plt.plot(x,y)
plt.show()

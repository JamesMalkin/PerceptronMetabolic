import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

df = pd.read_csv('total_metabolic.csv')
df2 = pd.read_csv('maintenance_metabolic.csv')
df3 = pd.read_csv('consolidation_metabolic.csv')
dfList = [df, df2, df3]
clfList = []

def createModel(df, degrees):
    x = list(df.columns)
    X = []
    for value in x:
        X.append([float(value)])
    X = np.array(X)

    y = []
    for col in df.columns:
        y.append([df[col].mean()])
    y = np.array(y)

    poly = PolynomialFeatures(degree=degrees)
    X = poly.fit_transform(X)
    predict = np.arange(0, 21, 0.1).reshape(-1, 1)
    P = poly.fit_transform(predict)

    clf = linear_model.LinearRegression()
    clf.fit(X, y)

    print(clf.score(X, y))
    print(clf.coef_)

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(0, 21, 0.1), clf.predict(P))
    ax.scatter(x, y)
    plt.show()

    def f(number):
        l = []
        for d in range(degrees + 1):
            l.append((number ** d) * int(clf.coef_[0][d]))
        return sum(l)

    print(optimize.minimize(f, x0=6))

    return clf.predict(P)

degrees = [3,1,3]
def set_degrees(degrees):
    degree = 0
    for df in dfList:
        clfList.append(createModel(df, degrees[degree]))
        degree += 1

set_degrees(degrees)

fig, ax = plt.subplots(1, 1)
#ax.plot(np.arange(0, 21, 0.1), clfList[0])
ax.plot(np.arange(0, 21, 0.1), clfList[1])
ax.plot(np.arange(0, 21, 0.1), clfList[2])
total = np.array(clfList[1]) + np.array(clfList[2])
ax.plot(np.arange(0, 21, 0.1), total)
plt.show()
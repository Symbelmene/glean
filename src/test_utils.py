import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generateSinWave(numPts, numPeriods, offset=0):
    x = np.linspace(0, numPeriods * 2 * np.pi, numPts)
    y = np.sin(x + offset)
    return y


def generateRandomTicker(numPts):
    y = generateSinWave(numPts, random.randint(1, 10), offset=random.randint(0, 6))
    for _ in range(random.randint(1, 5)):
        y = y + generateSinWave(numPts, random.randint(1, 10), offset=random.randint(0, 6))
    y = y + np.random.rand(numPts)
    return y - np.min(y) + random.randint(1, 200)


def generateRandomTickerDataframe(numPts, numTickers):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    arr = np.array([generateRandomTicker(numPts) for _ in range(numTickers)]).T
    df = pd.DataFrame(arr, columns=[3*alphabet[i] for i in range(numTickers)])
    return df

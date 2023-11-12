import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from tqdm import tqdm

import utils
from config import Config, Interval
cfg = Config()


class Stock:
    def __init__(self, name):
        self.name = name
        self.path = f'{cfg.DATA_DIR_24_HOUR}/{name}.csv'
        self.data = self.load()

    def load(self):
        try:
            df = pd.read_csv(self.path, index_col=0, parse_dates=['Date'])
            return df[~df.index.duplicated(keep='first')]
        except FileNotFoundError:
            print(f"ERROR: File {self.path} not found")
            exit(1)

    def slice(self, start=None, end=None):
        if not start:
            start = self.data.index[0]
        else:
            end = pd.to_datetime(end)
            if start < self.data.index[0]:
                raise ValueError(f"ERROR: The time slice on the dataframe for {self.name} "
                                    f"extends before the start of available data")
        if not end:
            end = self.data.index[-1]
        else:
            start = pd.to_datetime(start)
            if end > self.data.index[-1]:
                raise ValueError(f"ERROR: The time slice on the dataframe for {self.name} "
                                    f"extends beyond the end of available data")

        mask = (self.data.index >= start) & (self.data.index <= end)
        self.data = self.data[mask]


def buildLSTMModel(historySize, numHistoryVars, predictionSize):
    # Builds and Functional LSTM model using the windowSize and numStocks as input and numActions
    # as output
    inputLayer = layers.Input(shape=(historySize, numHistoryVars))
    x = layers.LSTM(64, activation='tanh', return_sequences=True)(inputLayer)
    x = layers.LSTM(64, activation='tanh')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(cfg.NEURAL_NETWORK.DROPOUT_RATE)(x)
    x = layers.Dense(predictionSize*2, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(predictionSize*2, activation='relu')(x)
    x = layers.Dense(predictionSize, activation='relu')(x)
    outputLayer = layers.Softmax()(x)
    model = tf.keras.Model(inputs=inputLayer, outputs=outputLayer)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def prepareTrainingDataset(historySize, predictionSize):
    # Choose numStocks random stocks from the datasets
    validTickers = utils.getValidTickers(Interval.DAY)
    stocks = [Stock(ticker) for ticker in random.sample(validTickers, 500)]

    keys = ['Open', 'High', 'Low', 'Close', 'Volume']
    xVals, yVals = [], []
    for stock in tqdm(stocks):
        df = stock.data
        arr = np.array(df[keys])[1:,:]

        # Slice the data in to windows of size windowSize
        for i in range(arr.shape[0] - (historySize + predictionSize)):
            xVal = arr[i:i+historySize]
            yVal = arr[i+historySize:i+historySize+predictionSize, 3]

            # Normalize the data
            xVal = (xVal - np.mean(xVal, axis=0)) / np.std(xVal, axis=0)
            yVal = (yVal - np.mean(yVal, axis=0)) / np.std(yVal, axis=0)

            xVals.append(xVal)
            yVals.append(yVal)

    # Shuffle the data
    xVals, yVals = np.array(xVals), np.array(yVals)
    indices = np.arange(xVals.shape[0])
    np.random.shuffle(indices)
    xVals, yVals = xVals[indices], yVals[indices]

    # Split the data into training and validation sets
    split = int(xVals.shape[0] * 0.8)
    xTrain, yTrain = xVals[:split], yVals[:split]
    xVal, yVal = xVals[split:], yVals[split:]

    # Save the data
    np.save(f'../train/xTrain.npy', xTrain)
    np.save(f'../train/yTrain.npy', yTrain)
    np.save(f'../train/xVal.npy', xVal)
    np.save(f'../train/yVal.npy', yVal)


def main():
    print(tf.config.list_physical_devices('GPU'))
    # Create dataset
    #prepareTrainingDataset(10, 5)

    # Load dataset
    xTrain = np.load(f'../train/xTrain.npy')
    yTrain = np.load(f'../train/yTrain.npy')
    xVal = np.load(f'../train/xVal.npy')
    yVal = np.load(f'../train/yVal.npy')

    # Build model
    model = buildLSTMModel(historySize=10,
                           numHistoryVars=5,
                           predictionSize=5)
    print(model.summary())

    # Train model
    model.fit(xTrain, yTrain, epochs=10, batch_size=256, validation_data=(xVal, yVal))



if __name__ == '__main__':
    main()
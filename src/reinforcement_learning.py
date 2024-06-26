import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import gymnasium as gym
from gymnasium import Env

import utils
from config import Config, Interval
cfg = Config()

if cfg.MODE == 'test':
    import test_utils


class StockMarket(Env):
    def __init__(self, numStocks, windowSize, start, end, startMoney, buyAmount=1000):
        super(StockMarket, self).__init__()

        # Define observation space: This will randomly select numStocks stocks from the dataset and create an
        # obervable window of size windowSize. The observation space will be a 3D array of shape (numStocks,
        # windowSize, 1) which will contain the Close value of the loaded stocks.
        self.observation_shape = (numStocks, windowSize, 1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float32)

        # Define action space. This is the number of different actions that can be taken by the agent, which is
        # buy or sell for each stock, or hold. This is a discrete space with 1+(numStocks*2) possible actions.
        self.action_space = gym.spaces.Discrete(1+(numStocks*2),)

        # Define chosen stocks in the environment to get information from
        self.tickers = []

        # Define the times to start and end the stock market simulation at
        self.start = start
        self.end = end

        # Define the starting money
        self.startMoney = startMoney

        # Store the buy amount
        self.buyAmount = buyAmount

    def reset(self):
        # Reset start money and holdings
        self.money = self.startMoney

        # Reset the reward
        self.totalReward = 0

        # Reset the current step
        self.currStep = 0

        if cfg.MODE == 'train':
            try:
                # Choose numStocks random stocks from the dataset
                validTickers = utils.getValidTickers(Interval.DAY)
                self.tickers = np.random.choice(validTickers, size=self.observation_shape[0], replace=False)
                stocks = [Stock(ticker) for ticker in self.tickers]
                self.holdings = {ticker : 0 for ticker in self.tickers}

                # Slice the ticker frames on the start and end dates then merge them into one dataframe
                for stock in stocks:
                    stock.slice(self.start, self.end)
            except ValueError: # TODO: This is a temporary fix for tickers with data missing in time period
                return self.reset()

            self.stockData = pd.concat([stock.data['Close'] for stock in stocks], axis=1, keys=self.tickers)
            self.stockData.fillna(method='ffill', inplace=True)
        elif cfg.MODE == 'test':
            df = test_utils.generateRandomTickerDataframe(252, self.observation_shape[0])
            self.tickers = df.columns
            self.stockData = df
            self.holdings = {ticker : 0 for ticker in self.tickers}

        self.window = self.stockData.iloc[:self.observation_shape[1]]

        returns = np.diff(self.stockData, axis=0) / np.array(self.stockData)[:-1,:]

        bestFractionalReturns = ((self.stockData.shift(1) / self.stockData) - 1).max(axis=1).apply(lambda v: max(v, 0))
        bestPossibleReturns = ((1 + bestFractionalReturns) * self.buyAmount)
        self.bestPossibleReturns = np.cumsum(bestPossibleReturns) + self.startMoney
        return self.window

    def get_action_meanings(self):
        actionList = ['Hold']
        for ticker in self.tickers:
            actionList.append(f'Buy {ticker}')
            actionList.append(f'Sell {ticker}')
        return {idx: action for idx, action in enumerate(actionList)}

    def step(self, action, verbose=False):
        # Carry out the action
        if action == 0:
            # Hold
            report = "Holding"
        elif action % 2 == 1:
            # Buy
            ticker = self.tickers[(action-1)//2]
            price = self.window.iloc[-1][ticker]
            if self.money >= self.buyAmount:
                self.money -= self.buyAmount
                numShares = self.buyAmount / price
                self.holdings[ticker] += numShares
                report = f"Bought {numShares} shares of {ticker} at {round(price, 3)}"
            else:
                report = f"ERROR: Not enough money to buy {ticker}"
        elif action % 2 == 0:
            # Sell
            ticker = self.tickers[(action-2)//2]
            price = self.window.iloc[-1][ticker]
            if self.holdings[ticker] > 0:
                self.money += price * self.holdings[ticker]
                self.holdings[ticker] = 0
                report = f"Sold {ticker} at {round(price, 3)}"
            else:
                report = f"ERROR: No {ticker} to sell"
        else:
            raise ValueError(f"ERROR: Invalid action {action}")

        # Calculate the reward
        bestPossible = self.bestPossibleReturns[self.currStep] # TODO: This needs fixing!
        currentReturn = 1 - self.money / self.startMoney
        if currentReturn > bestPossible:
            reward = 1
            print(f'WARNING: The current return {round(currentReturn, 3)} is greater than the best possible return {round(bestPossible, 3)}')
        elif bestPossible == 0:
            reward = 0
        else:
            reward = currentReturn / bestPossible

        # Increment the step counter
        self.currStep += 1

        # Get the new state window
        self.window = self.stockData.iloc[self.currStep:self.currStep+self.observation_shape[1]]

        # Determine if the episode is over
        done = False
        if self.currStep == len(self.stockData) - self.observation_shape[1]:
            done = True

        if verbose:
            self.summariseState(report)

        return self.window, reward, done, []

    def summariseState(self, report):
        print('\n')
        print('#' * 30)
        print(f"STEP {self.currStep}")
        print(report)
        totalShareValue = 0
        for idx, (ticker, shares) in enumerate(self.holdings.items()):
            if shares > 0:
                shareValue = round(shares*self.window.iloc[-1, idx], 2)
                shareStr = f'(${shareValue})'
                print(f"{ticker.ljust(5)}: {shares} {shareStr.rjust(8)} shares")
                totalShareValue += shareValue
        print('-'*20)
        print(f'Cash : ${round(self.money, 2)}')
        print(f'Net  : ${round(totalShareValue + self.money, 2)}')
        print('#' * 30)

    def render(self):
        pass


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


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * np.exp(-1. * current_step * self.decay)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, transition):
        # Push a transition to the memory
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.push_count % self.capacity] = transition
        self.push_count += 1

    def sample(self, batch_size):
        # Sample a batch of transitions from the memory
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        # Check if the memory contains enough transitions to provide a sample
        return len(self.memory) >= batch_size


class DQN_Agent:
    def __init__(self, strategy, numStocks, windowSize):
        self.strategy = strategy
        self.numStocks = numStocks
        self.windowSize = windowSize
        self.numActions = 2 * numStocks + 1 # Buy or sell per stock + hold
        self.currStep = 0

    def select_action(self, state, policyNet):
        rate = self.strategy.get_exploration_rate(self.currStep)
        self.currStep += 1

        if rate > random.random():
            return random.randrange(self.numActions), rate, True
        else:
            return np.argmax(policyNet.predict(np.array([normaliseState(state)]))), rate, False


def buildLSTMModel(numStocks, windowSize):
    # Builds a Functional LSTM model using the windowSize and numStocks as input and numActions
    # as output
    numActions = 2 * numStocks + 1
    inputLayer = layers.Input(shape=(windowSize, numStocks))
    x = layers.LSTM(cfg.NEURAL_NETWORK.LSTM_LAYER_1_SIZE, activation='tanh', return_sequences=True)(inputLayer)
    x = layers.LSTM(cfg.NEURAL_NETWORK.LSTM_LAYER_2_SIZE, activation='tanh')(x)
    x = layers.Dense(cfg.NEURAL_NETWORK.DENSE_LAYER_1_SIZE, activation='relu')(x)
    x = layers.Dropout(cfg.NEURAL_NETWORK.DROPOUT_RATE)(x)
    x = layers.Dense(numActions, activation='relu')(x)
    x = layers.Flatten()(x)
    outputLayer = layers.Softmax()(x)
    model = tf.keras.Model(inputs=inputLayer, outputs=outputLayer)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def copy_weights(copyFrom, copyTo):
    variables2 = copyFrom.trainable_variables
    variables1 = copyTo.trainable_variables
    for v1, v2 in zip(variables1, variables2):
        v1.assign(v2.numpy())


def normaliseState(state):
    # Normalise the state to be between 0 and 1
    return state / np.max(state, axis=0)


def normaliseStates(states):
    for i in range(states.shape[0]):
        states[i] = normaliseState(states[i])
    return states

import os
from enum import Enum

class Interval(Enum):
    DAY         = "1d"
    FIVE_MINUTE = "5m"


class Config:
    def __init__(self):
        self.docker = True if 'DOCKERCONTAINER' in os.environ else False

        self.env = parseEnvFile('../.env')

        self.THREADS   = 20
        self.PROCESSES = 8

        self.BASE_DIR = self.env['DB_DIR']
        self.TRAIN_DIR = self.BASE_DIR + '/train'

        self.DATA_DIR              = '/data' if self.docker else self.BASE_DIR + '/data'
        self.DATA_DIR_24_HOUR  = self.DATA_DIR + '/24_HOUR'
        self.DATA_DIR_5_MINUTE = self.DATA_DIR + '/5_MINUTE'

        self.PORT_DIR          = self.BASE_DIR + '/portfolios'

        self.LOG_DIR  = '/logs' if self.docker else self.BASE_DIR + '/logs'

        self.RISK_FREE_RATE = 0.0125

        self.REINFORCEMENT_LEARNING = RLConfig()
        self.STOCK_MARKET           = SMConfig()
        self.NEURAL_NETWORK         = NNConfig()


# REINFORCEMENT MODEL CONFIG #
class RLConfig:
    def __init__(self):
        # RL PARAMATERS #
        self.GAMMA = 0.99
        self.NUM_EPISODES   = 50
        self.STRATEGY_START = 1
        self.STRATEGY_END   = 0.001
        self.STRATEGY_DECAY = 0.001
        self.REPLAY_MEMORY_SIZE = 10000
        self.EPISODES_PER_NETWORK_UPDATE = 25


# STOCK MARKET SIMULATION PARAMETERS #
class SMConfig:
    def __init__(self):
        self.NUM_STOCKS  = 5
        self.WINDOW_SIZE = 10
        self.START_DATE  = '2019-01-01'
        self.END_DATE    = '2020-01-01'
        self.START_CASH  = 10000
        self.BUY_AMOUNT  = 1000


# NEURAL NETWORK PARAMETERS #
class NNConfig:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001

        self.LSTM_LAYER_1_SIZE  = 64
        self.LSTM_LAYER_2_SIZE  = 64

        self.DENSE_LAYER_1_SIZE = 32

        self.DROPOUT_RATE = 0.2


def parseEnvFile(envPath):
    if not os.path.exists(envPath):
        print(f'ERROR: No .env file found at {envPath}!')

    with open(envPath, 'r') as rf:
        envList = [val.split('=') for val in rf.read().split('\n')]
    envList = [e for e in envList if len(e) > 1]
    return {e[0] : e[1] for e in envList}

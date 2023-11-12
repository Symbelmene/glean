import wandb
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import namedtuple
from statistics import mean

import reinforcement_learning as rl

from config import Config
cfg = Config()

wandb.login()


def plotStockPricesToWandb(history):
    keys = list(history[0]['Stocks'].keys())

    stockVals = np.array([[history[i]['Stocks'][key] for i in range(len(history))] for key in keys])
    stockValsNorm = (stockVals - np.min(stockVals, axis=0)) / np.max(stockVals, axis=0)
    stockValsNormToList = [stockValsNorm[i].tolist() for i in range(len(keys))]

    try:
        plot = wandb.plot.line_series(
            xs=[i for i in range(len(history))],
            ys=stockValsNormToList,
            keys=keys,
            title="Stock Price Plots",
            xname="Days",
        )
    except:
        x = 1
    return plot


def getAssetValues(stockValues, holdings):
    return sum([stockValues[key] * holdings[key] for key in stockValues.keys()])


def plotMoneyAndAssetsToWandb(history):
    moneyList = [history[i]['Money'] for i in range(len(history))]
    assetsList = [getAssetValues(entry['Stocks'], entry['Assets']) for entry in history]
    plot = wandb.plot.line_series(
        xs=[i for i in range(len(history))],
        ys=[moneyList, assetsList, [moneyList[i] + assetsList[i] for i in range(len(history))] ],
        keys=['Money', 'Assets', 'Total'],
        title="Money",
        xname="Days",
    )
    return plot


def main():
    # Config values
    gamma = cfg.REINFORCEMENT_LEARNING.GAMMA

    episodesPerNetworkUpdate = cfg.REINFORCEMENT_LEARNING.EPISODES_PER_NETWORK_UPDATE
    numEpisodes = cfg.REINFORCEMENT_LEARNING.NUM_EPISODES
    # Initialise Environment
    sm = rl.StockMarket(numStocks=cfg.STOCK_MARKET.NUM_STOCKS,
                     windowSize=cfg.STOCK_MARKET.WINDOW_SIZE,
                     start=pd.to_datetime(cfg.STOCK_MARKET.START_DATE),
                     end=pd.to_datetime(cfg.STOCK_MARKET.END_DATE),
                     startMoney=cfg.STOCK_MARKET.START_CASH,
                     buyAmount=cfg.STOCK_MARKET.BUY_AMOUNT)

    run = wandb.init(
        project="stock-markey-rl",
        config={
            "learning_rate": cfg.NEURAL_NETWORK.LEARNING_RATE,
            "epochs": numEpisodes,
        })

    # Initialise classes
    strategy = rl.EpsilonGreedyStrategy(start=cfg.REINFORCEMENT_LEARNING.STRATEGY_START,
                                     end=cfg.REINFORCEMENT_LEARNING.STRATEGY_END,
                                     decay=cfg.REINFORCEMENT_LEARNING.STRATEGY_DECAY)
    agent = rl.DQN_Agent(strategy,
                      numStocks=cfg.STOCK_MARKET.NUM_STOCKS,
                      windowSize=cfg.STOCK_MARKET.WINDOW_SIZE)
    memory = rl.ReplayMemory(capacity=cfg.REINFORCEMENT_LEARNING.REPLAY_MEMORY_SIZE)

    # Experience Tuple
    Experience = namedtuple('Experience', ['states','actions', 'rewards', 'next_states', 'dones'])

    # Initialise Models & copy weights to ensure identical start
    policyNetwork = rl.buildLSTMModel(numStocks=cfg.STOCK_MARKET.NUM_STOCKS,
                                   windowSize=cfg.STOCK_MARKET.WINDOW_SIZE)
    targetNetwork = rl.buildLSTMModel(numStocks=cfg.STOCK_MARKET.NUM_STOCKS,
                                   windowSize=cfg.STOCK_MARKET.WINDOW_SIZE)
    rl.copy_weights(policyNetwork, targetNetwork)

    # Optimiser
    optimiser = tf.keras.optimizers.Adam(learning_rate=cfg.NEURAL_NETWORK.LEARNING_RATE)

    # Rewards
    totalRewards = np.empty(numEpisodes)

    for episode in range(numEpisodes):
        # Reset the environment
        state = sm.reset()
        episodeRewards = 0
        losses = []
        history = []
        actionDict = sm.get_action_meanings()

        # Run episode
        for timestep in itertools.count():
            action, rate, flag = agent.select_action(state, policyNetwork)
            nextState, reward, done, info = sm.step(action)

            # Add rewards
            episodeRewards += reward

            print(sm.holdings)
            history.append({"Action"      : actionDict[action],
                            "Rate"        : rate,
                            "Action Type" : 'explore' if flag else 'exploit',
                            "Stocks"      : state.iloc[-1].to_dict(),
                            "Money"       : sm.money,
                            "Assets"      : sm.holdings
                            })

            # Store the experience
            memory.push(Experience(state, action, nextState, reward, done))
            state = nextState

            if memory.can_provide_sample(batch_size=cfg.NEURAL_NETWORK.BATCH_SIZE):
                experiences = memory.sample(batch_size=cfg.NEURAL_NETWORK.BATCH_SIZE)
                batch = Experience(*zip(*experiences))
                states, actions, rewards, nextStates, dones = np.asarray(batch[0]), np.asarray(batch[1]), np.asarray(
                    batch[3]), np.asarray(batch[2]), np.asarray(batch[4])

                # Calculate TD-target
                qsaPrime = np.max(targetNetwork.predict(rl.normaliseStates(nextStates)), axis=1)
                qsaTarget = np.where(dones, rewards, rewards + gamma * qsaPrime)
                qsaTarget = tf.convert_to_tensor(qsaTarget, dtype='float32')

                # Back propagate loss
                with tf.GradientTape() as tape:
                    qsa = tf.math.reduce_sum(
                        policyNetwork(rl.normaliseStates(nextStates)) * tf.one_hot(actions, sm.action_space.n),
                        axis=1)
                    loss = tf.math.reduce_mean(tf.square(qsaTarget - qsa))

                # Update the policy network weights using ADAM optimiser
                variables = policyNetwork.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimiser.apply_gradients(zip(gradients, variables))

                losses.append(loss.numpy())
            else:
                losses.append(0)

            # If it is time to update target network
            if timestep % episodesPerNetworkUpdate == 0:
                rl.copy_weights(policyNetwork, targetNetwork)

            if done:
                break

            if done == True:
                break

        totalRewards[episode] = episodeRewards
        avg_rewards = totalRewards[max(0, episode - 100):(episode + 1)].mean()  # Running average reward of 100 iterations

        wandb.log({"Episode Reward": totalRewards[episode],
                   "Running Average Reward": avg_rewards,
                   "Losses": mean(losses),
                   "Stock Prices": plotStockPricesToWandb(history),
                   "Money & Assets": plotMoneyAndAssetsToWandb(history)
                   })

        if episode % 1 == 0:
            print(f"Episode:{episode} Episode_Reward:{totalRewards[episode]} "
                  f"Avg_Reward:{avg_rewards: 0.1f} Losses:{mean(losses): 0.1f} "
                  f"rate:{rate: 0.8f} flag:{flag}")


if __name__ == '__main__':
    main()
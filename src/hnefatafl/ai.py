"""AI for Hnefatafl. Uses the Hnefatafl library for the game logic."""
from functools import partial
from multiprocessing import Pool
import os
import pickle

import tensorflow as tf
import hnefatafl as hn
from hnefatafl.env import env
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from pettingzoo.test import api_test

env = env()  # create the environment
# api_test(env, num_cycles=10000, verbose_progress=True)  # test the environment

LR = 1e-3

def some_random_games():
    """Plays some random games to see how the environment works."""
    for _ in range(20):
        env.reset()
        observation, reward, terminated, truncated, info = env.last()
        for i, agent in enumerate(env.agent_iter()):
            action = env.action_space(agent).sample(mask=observation["action_mask"])
            env.step(action)
            observation, reward, terminated, truncated, info = env.last()
            if terminated:
                print(f"Agent {agent} won!")
                print(f"Game lasted {i} steps.")
                print(f"Game result: {info}")
                break

LEGAL_OUTCOMES = [
    hn.Outcome(hn.Termination.KING_CAPTURED, hn.BLACK),
    hn.Outcome(hn.Termination.KING_ESCAPED, hn.WHITE),
    hn.Outcome(hn.Termination.STALEMATE, None),
]

def initial_population(initial_games=10000, print_stats=False, print_every=100):
    """Generates the initial population of games to train on.
    Returns:
        training_data: An array of games, each game is a list of moves, each move is a tuple of
            (observation, action, reward).
        outcomes: A list of outcomes for each game."""
    training_data = []
    outcomes = []
    for i in range(initial_games):
        env.reset()
        observation, reward, terminated, truncated, info = env.last()
        game_memory = []
        prev_observation = observation
        for agent in env.agent_iter(max_iter=1000):
            action = env.action_space(agent).sample(mask=observation["action_mask"])
            env.step(action)
            observation, reward, terminated, truncated, info = env.last()
            """The observation is a (11x11x4) tensor, where the first two dimensions are the black and white
            pieces, and the third dimension is the king. The fourth dimension is all zeros if it is the
            black player's turn, and all ones if it is the white player's turn.
            The reward is 1 if the player won, -1 if the player lost, and 0 if the game is still going.
            The terminated variable is True if the game is over, and False if the game is still going.
            The truncated variable is True if the game ended in a stalemate, and False if the game is still going.
            The info variable is a dictionary containing the outcome of the game.
            The action is number between 0 and 14640, representing the move from a 11x11 board to a 11x11 board."""
            game_memory.append([prev_observation['observation'], action, reward])
            prev_observation = observation
            if terminated:
                assert env.unwrapped.board.outcome() in LEGAL_OUTCOMES
                outcomes.append(env.unwrapped.board.outcome())
                break
        # if the game ended in a stalemate, we don't want to use it for training
        if env.unwrapped.board.outcome() == hn.Outcome(hn.Termination.STALEMATE, None):
            continue
        training_data.append(game_memory)
        if i % print_every == 0:
            print(f"Game {i} of {initial_games} complete.")

    # print some statistics about the initial population
    if print_stats:
        print(Counter([outcome.result() for outcome in outcomes]))
        print(Counter([outcome.termination for outcome in outcomes]))
        print(Counter([hn.COLOR_NAMES[outcome.winner] for outcome in outcomes if outcome.winner is not None]))
        # average number of moves per game
        print(mean([len(game) for game in training_data]))
        # median number of moves per game
        print(median([len(game) for game in training_data]))
    return training_data, outcomes


def neural_network_model(input_size = 484) -> tflearn.DNN:
    """Creates a neural network model.
    
    Args:
        input_size: The size of the input layer. This is the number of features in the
            observation. The observation is a (11x11x4) tensor, so the input size is 484.
    Returns:
        network: The neural network model."""
    network = input_data(shape=[None, input_size, 1], name='input', dtype=tf.float32)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 14641, activation='softmax')
    """14641 is the number of possible actions (11**4), representing the probability of
    each action being taken."""

    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    return tflearn.DNN(network, tensorboard_dir='log')

def train_model(training_data, model=False):
    """Trains the neural network model.
    
    Args:
        training_data: An array of games, each game is a list of moves, each move is a tuple of
            (observation, action, reward).
        model: The neural network model. If False, creates a new model.
    Returns:
        model: The trained neural network model."""
    X = np.array([i[0] for game in training_data for i in game]).reshape(-1, 484, 1) 
    """Observations are 11x11x4 tensors flattened into a 484x1 vector for input into
    the neural network"""

    """Actions are integers from 0 to 14640 (11**4 - 1)"""
    y = [i[1] for game in training_data for i in game]

    # perform one-hot encoding on the actions
    y = tflearn.data_utils.to_categorical(y, 14641)

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')
    return model

# use multiprocessing to speed up training data generation
with Pool() as p:
    """I have 8 cores on my machine, so I use 8 processes."""
    results = p.map(initial_population, [10000]*8)  # 10000 games per process

# combine the results from each process
training_data = []
outcomes = []
for result in results:
    training_data.extend(result[0])
    outcomes.extend(result[1])

# save the initial population to a file
with open("training_data.pickle", "wb") as f:
    pickle.dump(training_data, f)

# load the initial population from a file
with open("training_data.pickle", "rb") as f:
    training_data = pickle.load(f)

model = train_model(training_data)
# save the trained model to a file
model.save("model.tflearn")

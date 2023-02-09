import os
import time
from typing import Tuple

import numpy as np
from tensorflow.python.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Model

from alphazero.neuralnet import NeuralNet
from alphazero.utils import *


class TaflNNet:
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(
            shape=(self.board_x, self.board_y, 4)
        )  # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 4))(
            self.input_boards
        )  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same", use_bias=False)(x_image)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same", use_bias=False)(h_conv1)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="valid", use_bias=False)(h_conv2)
            )
        )  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="valid", use_bias=False)(h_conv3)
            )
        )  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))
            )
        )  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))
            )
        )  # batch_size x 1024
        self.pi = Dense(self.action_size, activation="softmax", name="pi")(
            s_fc2
        )  # batch_size x self.action_size
        self.v = Dense(1, activation="tanh", name="v")(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer="adam",
        )


args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": False,
        "num_channels": 512,
    }
)


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = TaflNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(self, board) -> Tuple[np.array, float]:
        """Runs a prediction on a single board

        Arguments:
            board {np.array} -- observation of the board state (11, 11, 4)

        Returns:
            (np.array, float) -- policy vector, value
        """
        # preparing input
        board = board[np.newaxis, :, :]
        # run
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise f"No model in path {filepath}"
        self.nnet.model.load_weights(filepath)

"""MuZero configuration for Hnefatafl."""
from datetime import datetime
import pathlib
from typing import List, Literal, Tuple
from muzero_baseline.games.abstract_game import AbstractGame
import numpy as np
import torch

from hnefatafl import BoardT, KingEscapeAndCaptureEasierBoard, Move
from hnefatafl.utils import get_observation, move_to_action, action_to_move


class MuZeroConfig:
    def __init__(self) -> None:
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1

        """The following parameters are specific to the game of Hnefatafl."""
        self.observation_shape = (4, 11, 11)
        """Dimensions of the game observation, must be 3D: (channel, width, height)"""
        self.action_space = list(range(121 * 2 * 11))
        """Fixed list of all possible actions. Actions are defined as integers which are converted to moves.

        For Hnefatafl, a move is encoded as follows:
        1. Choose a piece on the board to move (0-120)
        2. Choose a direction to move the piece [HORIZONTAL, VERTICAL] (0-1)
        3. Choose which square on the line to move the piece to (0-10)

        The action space is therefore 121 * 2 * 11 = 2662 actions, with 242 "null" actions consisting of moving a piece
        to the square it is already on.
        """
        self.players = [0, 1]
        """List of players. In Hnefatafl, there are two players: 0 (attacker) and 1 (defender)"""
        self.stacked_observations = 0
        """Number of previous observations and previous actions to add to the current observation.

        For Hnefatafl, this is set to 0 because the game is fully observable."""

        self.muzero_player = 0
        """Turn Muzero begins to play (0: MuZero plays attacker, 1: MuZero plays defender)"""
        self.opponent = "self"

        """Parameters for the self-play"""
        self.num_workers = 1
        """Number of simultaneous threads/workers self-playing to feed the replay buffer"""
        self.selfplay_on_gpu = False
        """Whether or not to use the GPU if available during self-play"""
        self.max_moves = 1000
        """Maximum number of moves if game is not finished before"""
        self.num_simulations = 50
        """Number of future moves self-simulated"""
        self.discount = 1
        """Discount factor for the reward. Set to 1 for board games with a single reward at the end of the game."""
        self.temperature_threshold = None
        """Number of moves before dropping temperature to 0 (ie playing according to the max). If None,
        temperature is not dropped."""

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"
        """Choose the neural network used for the value and policy prediction:
        "resnet" for a residual network, "fullyconnected" for a linear multi-layer perceptron."""
        self.support_size = 10
        """Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size."""

        # Residual Network
        self.downsample = False
        """Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)"""
        self.blocks = 8
        """Number of blocks in the ResNet"""
        self.channels = 256
        """Number of channels in the ResNet"""
        self.reduced_channels_reward = 4
        """Number of channels in reward head"""
        self.reduced_channels_value = 4
        """Number of channels in value head"""
        self.reduced_channels_policy = 4
        """Number of channels in policy head"""
        self.resnet_fc_reward_layers = [128]
        """Define the hidden layers in the reward head of the dynamic network"""
        self.resnet_fc_value_layers = [128]
        """Define the hidden layers in the value head of the prediction network"""
        self.resnet_fc_policy_layers = [128]
        """Define the hidden layers in the policy head of the prediction network"""

        # Fully Connected Network
        self.encoding_size = 484
        """Number of channels (feature maps) in the representation network. For hnetafl, this is 4 * 11 * 11 = 484 channels."""
        self.fc_representation_layers = []
        """Define the hidden layers in the representation network"""
        self.fc_dynamics_layers = [256]
        """Define the hidden layers in the dynamics network"""
        self.fc_reward_layers = [256]
        """Define the hidden layers in the reward network"""
        self.fc_value_layers = []
        """Define the hidden layers in the value network"""
        self.fc_policy_layers = []
        """Define the hidden layers in the policy network"""

        ### Training
        self.results_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        """Path to store the model weights and TensorBoard logs"""
        self.save_model = True
        """Save the model every checkpoint_interval in results_path as model.checkpoint"""
        self.training_steps = 20000
        """Total number of training steps (ie weights update according to a batch)"""
        self.batch_size = 1024
        """Number of parts of games to train on at each training step"""
        self.checkpoint_interval = 1000
        """Number of training steps before using the model for self-playing"""
        self.value_loss_weight = 0.25
        """Scale the value loss to avoid overfitting of the value function, paper recommends 0.25"""
        self.train_on_gpu = torch.cuda.is_available()
        """Whether or not to use the GPU if available during training"""
        self.optimizer: Literal["SGD", "Adam"] = "Adam"
        """Which optimizer to use for training, "Adam" or "SGD". Paper uses SGD."""
        self.weight_decay = 1e-4
        """L2 weights regularization"""
        self.momentum = 0.9
        """Used only if optimizer is SGD"""

        # Exponential learning rate schedule
        self.lr_init = 0.002
        """Initial learning rate"""
        self.lr_decay_rate = 0.9
        """Set it to 1 to use a constant learning rate"""
        self.lr_decay_steps = 10000
        """Number of steps before decreasing the learning rate"""

        # Replay Buffer
        self.replay_buffer_size = 100000
        """Number of self-play games to keep in the replay buffer"""
        self.num_unroll_steps = 1000
        """Number of game moves to keep for every batch element"""
        self.td_steps = 1000
        """Number of steps in the future to take into account for calculating the target value"""
        self.PER = False
        """Prioritized Experience Replay (See paper appendix Training)

        If True, select in priority the elements in the replay buffer which are unexpected for the network."""
        self.PER_alpha = 0.6
        """How much prioritization is used, 0 corresponding to the uniform case, and 1 to the original case."""

        # Reanalyze
        self.use_last_model_value = False
        """Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyse)"""
        self.reanalyse_on_gpu = False
        """Whether or not to use the GPU if available during reanalysis"""

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0
        """Number of seconds to wait after each played game"""
        self.training_delay = 0
        """Number of seconds to wait after each training step"""
        self.ratio = 1
        """Desired training steps per self played game ratio.
        
        Set to None to disable the ratio adjustment."""

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """Parameter to alter the visit count distribution to ensure that the action selection
        becomes greedier as training progresses. The smaller it is, the more likely the best action
        (with the highest visit count) is to be selected.

        Args:
            trained_steps: Number of training steps performed so far.

        Returns:
            A positive float."""
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class HnefataflGame(AbstractGame):
    """Game wrapper"""

    def __init__(self, seed: int = None):
        """Initialize the game.

        Args:
            seed: Seed for the game.
        """
        self.env = HnefEnv()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Apply action to the game.

        Args:
            action: Action to apply.
        Returns:
            The new observation, the reward, and a boolean indicating whether the game is over.
        """
        return self.env.step(action)

    def to_play(self):
        """Return the current player."""
        return self.env.to_play()

    def legal_actions(self) -> List[int]:
        """Return the list of legal actions."""
        return self.env.legal_actions()

    def reset(self):
        """Reset the game for a new game."""
        return self.env.reset()

    def close(self):
        """Close the game."""
        pass

    def render(self, **kwargs):
        """Display the game."""
        self.env.render()

    def human_to_action(self) -> int:
        """For multiplayer games, ask the user for a legal move and return the corresponding action number."""
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action: int) -> str:
        """Return the string corresponding to an action."""
        return self.env.action_to_human_input(action)


class HnefEnv:
    """Environment for Hnefatafl, as used in HnefataflGame."""

    def __init__(
        self, game_type: BoardT = KingEscapeAndCaptureEasierBoard
    ):
        self.board = game_type()
        self.board.move_limit = 1000
        self.player = -1  # -1 is black, 1 is white

    def to_play(self):
        """Return the current player."""
        return 0 if self.player == -1 else 1
    
    def reset(self):
        """Reset the game for a new game."""
        self.board.reset()
        self.player = -1
        return self.get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Apply action to the game.

        Args:
            action: Action to apply.

        Returns:
            The new observation, the reward, and a boolean indicating whether the game is over.
        """
        move = action_to_move(action)
        self.board.push(move)
        done = self.board.is_game_over()
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            reward = 0
        elif outcome.winner == self.player < 0:  # Black wins (player is black)
            reward = 1
        elif outcome.winner == self.player > 0:  # White wins (player is white)
            reward = 1
        else:
            reward = -1
        self.player *= -1
        return self.get_observation(), reward, done
    
    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        return get_observation(self.board, self.to_play())
    
    def legal_actions(self) -> List[int]:
        """Return the list of legal actions."""
        return [move_to_action(move) for move in self.board.legal_moves]
    
    def render(self, **kwargs):
        """Display the game."""
        print(self.board)

    def human_input_to_action(self) -> Tuple[bool, int]:
        """Ask the user for a legal move and return the corresponding action number."""
        move = input("Enter a move: ")
        try:
            move = Move.from_code(move)
        except ValueError:
            print("Invalid move")
            return False, None
        if move not in self.board.legal_moves:
            print("Illegal move")
            return False, None
        return True, move_to_action(move)
    
    def action_to_human_input(self, action: int) -> str:
        """Return the string corresponding to an action."""
        return action_to_move(action).code()

"""The main module for the Hnefatafl game. This module contains the
:class:`Game` class, which is responsible for managing the game.
"""

from hnefatafl import KingEscapeAndCaptureEasierBoard, BLACK, WHITE
from hnefatafl.game import Game, HumanPlayer, RandomPlayer, GreedyWithKingPlayer
from hnefatafl.utils import prettify_board

def play_humans():
    """Play a game of Hnefatafl."""
    # Create the board.
    board = KingEscapeAndCaptureEasierBoard()
    # Create the players.
    players = [HumanPlayer(BLACK), HumanPlayer(WHITE)]
    # Create the game.
    game = Game(board, players)
    # Play the game.
    game.play()


def play_random():
    """Play a game of Hnefatafl against a player that plays random moves."""
    # Create the board.
    board = KingEscapeAndCaptureEasierBoard()
    # Create the players.
    players = [HumanPlayer(BLACK), RandomPlayer(WHITE)]
    # Create the game.
    game = Game(board, players)
    # Play the game.
    game.play()


def play_greedy():
    """Play a game of Hnefatafl against a player that plays greedy moves (always captures if possible)."""
    # Create the board.
    board = KingEscapeAndCaptureEasierBoard
    # Create the players.
    players = [HumanPlayer(BLACK, outputfn=prettify_board), GreedyWithKingPlayer(WHITE)]
    # Create the game.
    game = Game(board, *players, outputfn=prettify_board)
    # Play the game.
    game.play()


if __name__ == '__main__':
    play_greedy()
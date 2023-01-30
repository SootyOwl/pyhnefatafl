import numpy as np
import pytest
from hnefatafl.board import Board, Game, Piece, Position, PieceType


@pytest.fixture
def board():
    return Board()


@pytest.fixture
def board_starting_position(board):
    board.initialize_board()
    return board


@pytest.fixture
def board_king_captured(board):
    one_move_from_cap = np.array(
        [  #  A    B    C    D    E    F    G    H    I    J    K
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 1
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 2
            [" ", " ", " ", " ", "A", " ", " ", " ", " ", " ", " "],  # 3
            [" ", " ", " ", "A", "K", "A", " ", " ", " ", " ", " "],  # 4
            [" ", " ", " ", " ", " ", "A", " ", " ", " ", " ", " "],  # 5
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 6
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 7
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 8
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 9
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 10
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 11
        ]
    )
    board.state = one_move_from_cap
    return board


@pytest.fixture
def board_king_escaped(board):
    one_move_from_escape = np.array(
        [  #  A    B    C    D    E    F    G    H    I    J    K
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 1
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 2
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 3
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 4
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 5
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 6
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 7
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 8
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 9
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 10
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", "K", " "],  # 11
        ]
    )
    board.state = one_move_from_escape
    return board


def test_board_creation(board_starting_position: Board):
    board: Board = board_starting_position
    assert board.state.shape == (11, 11)
    assert board.get_piece("D1") == Piece(Position("D1"), PieceType.ATTACKER)
    assert board.get_king_location() == Position("F6")


def test_get_possible_moves(board_starting_position: Board):
    board: Board = board_starting_position
    moves = board.get_possible_moves(board["A5"])
    assert len(moves) == 3
    # only valid moves for A5 at the start of the game are B5, C5, and D5
    assert Position("B5") in moves
    assert Position("C5") in moves
    assert Position("D5") in moves


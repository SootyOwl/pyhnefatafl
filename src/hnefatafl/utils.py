"""Utils for the hnefatafl package."""
from functools import lru_cache
import numpy as np
import hnefatafl as hn



def prettify_board(board: hn.BoardT) -> str:
    """Output the board in a pretty format.
    
    Replaces the simple ascii representation with a more readable one.
    """
    # get the board's string representation
    board_str = str(board)
    # replace the empty squares with an empty square symbol
    board_str = board_str.replace(".", "·")
    # replace the pieces
    board_str = board_str.replace("m", "♙")
    board_str = board_str.replace("M", "♟")
    # black king
    board_str = board_str.replace("K", "♚")
    # fix the K column
    board_str = board_str.replace("♚", "K", 1)
    return board_str

def get_observation(board: hn.BoardT, player: hn.Color) -> np.ndarray:
    """Returns the observation of the board.

    This is a 4x11x11 matrix where the first two dimensions are the board
    (black men are in the first channel, white men are in the second
    channel), the third channel is the king position, and the fourth channel
    is for storing current turn color. In the case of hnefatafl, 
    the fourth channel's is the player color (0 for black, 1 for white).

    Use bitwise operations to extract the information you need for the pieces.
    """
    black_pieces = board.occupied_co[hn.BLACK]  # bitboard of black pieces
    white_pieces = board.occupied_co[hn.WHITE]  # bitboard of  white pieces

    # get the king position
    king_position = board.kings  # bitboard of the king position
    # get the squareset
    king_position = hn.SquareSet(king_position)
    black_pieces = hn.SquareSet(black_pieces)
    white_pieces = hn.SquareSet(white_pieces)

    # create the observation
    observation = np.zeros((4, 11, 11), dtype=np.int8)
    # set the black pieces
    for square in black_pieces:
        observation[0, hn.square_rank(square), hn.square_file(square)] = 1
    # set the white pieces
    for square in white_pieces:
        observation[1, hn.square_rank(square), hn.square_file(square)] = 1
    # set the king position
    for square in king_position:
        observation[2, hn.square_rank(square), hn.square_file(square)] = 1

    # the final channel is the turn indicator, which is the player color
    turn_indicator = 0 if player == hn.BLACK else 1
    observation[3, :, :] = turn_indicator

    return observation


def get_board(
    observation: np.ndarray, board_class=hn.KingEscapeAndCaptureEasierBoard
) -> hn.BoardT:
    """Returns a board from the observation."""
    # get the player color
    turn_indicator = observation[:, :, 3][0, 0]
    assert np.all(turn_indicator == 0) or np.all(turn_indicator == 1)
    player2move = hn.BLACK if turn_indicator == 0 else hn.WHITE

    # get the king position
    king_position = observation[:, :, 2]
    king_position = np.where(king_position == 1)
    assert len(king_position[0]) == 1 and len(king_position[1]) == 1
    king_position = hn.square(king_position[1][0], king_position[0][0])

    black_pieces = _get_pieces_from_observation(observation, 0)
    white_pieces = _get_pieces_from_observation(observation, 1)
    # create the board
    board = board_class(code=None)
    # create a boardstate object
    board_state = hn._BoardState(board)
    # create squaresets for the pieces
    black_pieces = int(hn.SquareSet(black_pieces))
    white_pieces = int(hn.SquareSet(white_pieces))
    king_position = int(hn.SquareSet([king_position]))
    board.men = black_pieces | white_pieces & ~king_position

    board_state.occupied_b = black_pieces
    board_state.occupied_w = white_pieces | king_position

    board_state.kings = king_position
    board_state.occupied = board_state.occupied_b | board_state.occupied_w
    board_state.turn = player2move
    # set the board state
    board_state.restore(board)
    return board


def _get_pieces_from_observation(observation, white=False) -> list:
    result = observation[:, :, white]
    result = np.where(result == 1)
    result = [hn.square(file, rank) for file, rank in zip(result[1], result[0])]

    return result


def board_from_code(code: str) -> hn.BoardT:
    """Returns a board from the code."""
    board = hn.Board()
    board._set_board_code(code)
    return board


@lru_cache(maxsize=10000)
def action_to_move(action: int) -> hn.Move:
    """Converts the action (an int representing the move id in a flattened 121x2x11 array)
    where the first number represent the starting position square, the third dimension is 0
    if the piece is moving along the file axis and 1 if the piece is moving along the rank axis,
    and the fourth dimension is the destination square on the line of movement (file[0-10] or rank[0-10])
    """
    # the action is the index of the 1 in the flattened array, so we can use np.unravel_index to get the
    # starting square, axis, and destination square
    from_square, axis, line_id = np.unravel_index(action, (121, 2, 11))

    # convert the axis and line_id to a destination square
    if axis == 0:  # file is the same as the starting square
        to_square = hn.square(hn.square_file(from_square), line_id)
    else:  # rank is the same as the starting square
        to_square = hn.square(line_id, hn.square_rank(from_square))

    return hn.Move(from_square, to_square)


# cache the moves
@lru_cache(maxsize=100000)
def move_to_action(move: hn.Move) -> int:
    """Converts the move to an action (an int representing the move index in a flattened 121x2x11 array)
    where the first two dimensions reqpresent the starting position, the third dimension is 0
    if the piece is moving along the file axis and 1 if the piece is moving along the rank axis,
    and the fourth dimension is the destination square on the line of movement (file[0-10] or rank[0-10])
    """
    if hn.square_file(move.from_square) == hn.square_file(move.to_square):
        axis = 0  # file movement
    else:
        axis = 1  # rank movement

    # get the destination square id (0-10)
    line_id = (
        hn.square_rank(move.to_square) if axis == 0 else hn.square_file(move.to_square)
    )

    assert line_id >= 0 and line_id <= 10

    # calculate the action
    action_zero = np.zeros((121, 2, 11), dtype=np.int8)
    action_zero[move.from_square, axis, line_id] = 1
    action = np.where(action_zero.flatten() == 1)
    assert len(action[0]) == 1
    return action[0][0]


def result_to_int(result: str) -> int:
    """Converts the result to an integer.

    -1 if black wins, 1 if white wins, 0 if draw.
    """
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    elif result == "1/2-1/2":
        return 0
    else:
        assert False, "bad result"


import sys, os, random, time, warnings
from hashlib import sha1

import numpy as np
from numpy import all, array, uint8


def sample(probs, T=1.0):
    keys = list(probs.keys())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = np.exp(np.array([probs[key] for key in keys]) / T).clip(0, 1e9)
    return keys[np.random.choice(len(keys), p=values / values.sum())]


class hashable(object):
    """Hashable wrapper for ndarray objects.
    Instances of ndarray are not hashable, meaning they cannot be added to
    sets, nor used as keys in dictionaries. This is by design - ndarray
    objects are mutable, and therefore cannot reliably implement the
    __hash__() method.
    The hashable class allows a way around this limitation. It implements
    the required methods for hashable objects in terms of an encapsulated
    ndarray object. This can be either a copied instance (which is safer)
    or the original object (which requires the user to be careful enough
    not to modify it).
    """

    def __init__(self, wrapped, tight=False):
        """Creates a new hashable object encapsulating an ndarray.
        If the "tight" parameter is True, a copy of the wrapped ndarray is
        created and encapsulated. Otherwise, the wrapped ndarray itself is
        encapsulated.

        Parameters
        ----------
        wrapped : ndarray
            The ndarray to encapsulate.
        tight : bool, optional
            Whether to create a copy of the wrapped ndarray. Defaults to
            False.
        """
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return repr(self.__wrapped)

    def unwrap(self):
        """Returns the encapsulated ndarray.
        If the wrapper is "tight", a copy of the encapsulated ndarray is
        returned. Otherwise, the encapsulated ndarray itself is returned.
        """
        return array(self.__wrapped) if self.__tight else self.__wrapped


if __name__ == "__main__":
    # test observation to board and back
    board = board_from_code(hn.STARTING_POSITION_CODE.split()[0])
    observation = get_observation(board, board.turn)
    board2 = get_board(observation)
    assert board == board2, f"\n{board}\n!=\n{board2}"

    # make 20 random moves
    for _ in range(20):
        move = random.choice(list(board.legal_moves))
        board.push(move)
    observation = get_observation(board, board.turn)
    board2 = get_board(observation)
    assert board == board2, f"\n{board}\n!=\n{board2}"

    # test action to move and back
    move = hn.Move.from_code("A1.2")
    action = move_to_action(move)
    move2 = action_to_move(action)
    assert move == move2, f"\n{move}\n!=\n{move2}"

    # test move to action and back
    for action in range(121 * 2 * 11):
        move = action_to_move(action)
        if not move:
            print(action)
            continue
        action2 = move_to_action(move)
        assert action == action2, f"\n{action}\n!=\n{action2}"

    # test action masking
    board = board_from_code(
        code="3mmmmm3/5m5/11/m4M4m/1m2MMM3m/mm1MMKMM1mm/m3MMM3m/m4M1m2m/11/5m5/3mmmm4"
    )

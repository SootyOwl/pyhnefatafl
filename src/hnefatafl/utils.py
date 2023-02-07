"""Utils for the hnefatafl package."""
from functools import lru_cache
import numpy as np
import hnefatafl as hn

def get_observation(board: hn.BoardT, player: hn.Color) -> np.ndarray:
    """Returns the observation of the player.
    
    This is a 11x11x4 matrix where the first two dimensions are the board
    (friendly pieces are in the first channel, enemy pieces are in the second
    channel), the third channel is the king position, and the fourth channel
    is the player color (all zeros for black, all ones for white)
    Use bitwise operations to extract the information you need.
    """
    if player == hn.BLACK:
        friendly_pieces = board.occupied_co[hn.BLACK]  # bitboard of friendly pieces
        enemy_pieces = board.occupied_co[hn.WHITE]  # bitboard of enemy pieces
    else:
        friendly_pieces = board.occupied_co[hn.WHITE]  # bitboard of friendly pieces
        enemy_pieces = board.occupied_co[hn.BLACK]  # bitboard of enemy pieces

    # get the king position
    king_position = board.kings  # bitboard of the king position
    # get the squareset
    king_position = hn.SquareSet(king_position)
    friendly_pieces = hn.SquareSet(friendly_pieces)
    enemy_pieces = hn.SquareSet(enemy_pieces)

    # create the observation
    observation = np.zeros((11, 11, 3), dtype=np.int8)
    # set the friendly pieces
    for square in friendly_pieces:
        observation[hn.square_rank(square), hn.square_file(square), 0] = 1
    # set the enemy pieces
    for square in enemy_pieces:
        observation[hn.square_rank(square), hn.square_file(square), 1] = 1
    # set the king position
    for square in king_position:
        observation[hn.square_rank(square), hn.square_file(square), 2] = 1

    # we need to include the player color in the observation for the neural network
    # to be able to distinguish between the two players (black and white)
    # we do this by adding a 4th channel to the observation, which is all zeros for black
    # and all ones for white
    if player == hn.BLACK:
        player_color = np.zeros((11, 11, 1), dtype=np.int8)
    else:
        player_color = np.ones((11, 11, 1), dtype=np.int8)

    observation = np.concatenate((observation, player_color), axis=2)
    return observation

@lru_cache(maxsize=100000)
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
    # by checking if the file or rank of the starting square is the same as the destination square
    if hn.square_file(move.from_square) == hn.square_file(move.to_square):
        axis = 0
    else:
        axis = 1

    # get the destination square id (0-10)
    line_id = hn.square_rank(move.to_square) if axis == 0 else hn.square_file(move.to_square)

    # calculate the action
    action_zero = np.zeros((121, 2, 11), dtype=np.int32)
    action_zero[move.from_square, axis, line_id] = 1
    return np.argmax(action_zero)


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


if __name__ == "__main__":
    # generate all *possible* moves and convert them to actions, then convert them back to moves and check that they are the same
    # files are a-k, ranks are 1-10
    action_to_move(11)

    moves = []
    non_null_moves = []
    for i in range(2662):
        move = action_to_move(i)
        moves.append(move)
        if not move:  # null move, part of the action space but just corresponds to moving to the same square as the starting square so we can ignore it
            continue
        action = move_to_action(move)
        assert i == action, f"{i}, move: {move}, action: {action}"
        non_null_moves.append(move)

    print(len(moves))  # 2662 including null moves
    print(len(non_null_moves))  # 2420 excluding the 242 null moves
    assert len(set(non_null_moves)) == len(non_null_moves), "there are duplicate non-null moves in the action space"
    null_move_indexes = {moves.index(move) for move in moves if not move}
    print(sorted(null_move_indexes))
    # this tells us that the null moves are at multiples of 22, so we can ignore them
    # [0, 22, 44, 66, 88, 110, 132, 154, 176, 198, 220, ..., 2650]
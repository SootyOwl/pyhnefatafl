"""Utils for the hnefatafl package."""
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
    player_color = np.zeros((11, 11), dtype=np.int8)
    """The player color of the agent."""
    player_color[:, :, 0] = player  # 0 if player 1, 1 if player 2
    observation = np.concatenate((observation, player_color), axis=2)
    return observation

def action_to_move(action: int) -> hn.Move:
    """Converts the action (an int representing the move id in a flattened 11x11x11x11 array)
    where the first two dimensions are the starting file and rank of the piece, the second two dimensions
    are the ending file and rank of the piece, to a move.
    """
    start_rank = action // (11*11*11)
    start_file = (action - start_rank*11*11*11) // (11*11)
    end_rank = (action - start_rank*11*11*11 - start_file*11*11) // 11
    end_file = action - start_rank*11*11*11 - start_file*11*11 - end_rank*11
    start_square = hn.square(start_file, start_rank)
    end_square = hn.square(end_file, end_rank)
    move = hn.Move(start_square, end_square)
    return move


def move_to_action(move: hn.Move) -> int:
    """Converts the move to an action (an int representing the move index in a flattened 11x11x11x11 array)
    where the first two dimensions are the starting file and rank of the piece, the second two dimensions
    are the ending position of the piece.
    """
    start_square = move.from_square
    end_square = move.to_square
    start_rank = hn.square_rank(start_square)
    start_file = hn.square_file(start_square)
    end_rank = hn.square_rank(end_square)
    end_file = hn.square_file(end_square)
    action = start_rank*11*11*11 + start_file*11*11 + end_rank*11 + end_file
    return action


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
    # create a board and get the observation
    board = hn.Board()
    observation = get_observation(board, hn.BLACK)
    print(observation["observation"].shape)
    print(observation["action_mask"].shape)

    move = hn.Move(hn.A1, hn.K1)
    print(move)
    action = move_to_action(move)
    print(action)
    move = action_to_move(action)
    print(move)

    assert move == hn.Move(hn.A1, hn.K1), f"{move} != {hn.Move(hn.A1, hn.K1)}"

    print(action_to_move(12689))
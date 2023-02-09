"""Implement the alphazero game protocol to allow the AI to play Hnefatafl."""

from typing import List, Tuple

import numpy as np

import hnefatafl as hn
import hnefatafl.utils as hn_utils
import itertools

# decorator to handle converting the numpy array to a board object
def array_to_board(func):
    def wrapper(self, board: np.ndarray, *args, **kwargs):
        if isinstance(board, np.ndarray):
            board = hn_utils.get_board(board)
        return func(self, board, *args, **kwargs)
    return wrapper


class HnefataflGame:
    """A wrapper for the Hnefatafl env to implement the Game protocol.

    Required by the alphazero framework.

    Implements:
        Game
    Methods:
        getInitBoard
        getBoardSize
        getActionSize
        getNextState
        getValidMoves
        getGameEnded
        getCanonicalForm
        getSymmetries
        stringRepresentation
    """

    def __init__(self, code=hn.Board().board_code(), move_limit=200):
        super().__init__()
        self.move_limit = move_limit
        self.board = hn.Board(code=code, move_limit=move_limit)

    def getInitBoard(self) -> np.ndarray:
        """Get the initial board."""
        self.board.reset_board()
        return hn_utils.get_observation(self.board, player=self.board.turn)

    def getBoardSize(self) -> tuple:
        """Get the board size."""
        return (11, 11)

    @array_to_board
    def getValidMoves(self, board: hn.BoardT, player) -> np.ndarray:
        """Get a list of all valid moves."""
        # player should actually be a bool
        legal_moves = board.legal_moves
        action_mask = np.zeros(2662, "int8")
        if not list(legal_moves):
            action_mask[-1] = 1
            return action_mask
        for i in legal_moves:
            action = hn_utils.move_to_action(i)
            assert i == hn_utils.action_to_move(action)
            action_mask[action] = 1
        return action_mask

    def getActionSize(self) -> int:
        """Get the number of all possible actions."""
        return 2662

    @array_to_board
    def getNextState(
        self, board: hn.BoardT, player, action: int
    ) -> Tuple[hn.BoardT, hn.Color]:
        """Get the next state of the board after applying the action.

        Args:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)"""
        move = hn_utils.action_to_move(action)
        # make a copy of the board
        next_board = board.copy()
        next_board.push(move)
        return self.getCanonicalForm(next_board, player), -player

    @array_to_board
    def getGameEnded(self, board: hn.BoardT, player) -> int:
        """Get the game result.

        Args:
            board: current board
            player: current player (1 or -1)

        Returns:
            1 if player won, -1 if player lost, 0 otherwise"""
        # player 1 is black, player -1 is white
        if board.is_game_over():
            # result_to_int returns -1 if black won, 1 if white won, 0 otherwise
            r2i = hn_utils.result_to_int(board.result())
            # if player is black, return 1 if black won, -1 if white won
            return -r2i if player == 1 else r2i
        return 0

    @array_to_board
    def getCanonicalForm(self, board: hn.BoardT, player) -> np.ndarray:
        """Get the canonical form of the board.

        Args:
            board: current board
            player: current player (1 or -1) (ignored)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. In hn, this is
                            achieved by returning the same observation if the
                            player is black and the inverted observation if the
                            player is white. The observation is inverted by
                            swapping the friendly and enemy piece arrays
                            and changing the player to move."""
        return hn_utils.get_observation(board, player=board.turn)

    @array_to_board
    def getSymmetries(
        self, board: hn.BoardT, pi: np.ndarray
    ) -> List[Tuple[hn.BoardT, np.ndarray]]:
        """Get symmetries of the board.

        Args:
            board: current board
            pi: policy vector for the current board

        Returns:
            symmetries: a list of (board,pi) tuples where each tuple is a
                        symmetry of the board and the corresponding pi vector.
                        This is used while training the neural network.
                        The length of the list of symmetries should be the
                        number of symmetries of the game board, not including
                        the original board."""
        ## mirror x2, rotational x4
        # assert len(pi) == 2662  # 11x11 board with 2662 moves
        # pi_board = np.reshape(pi, (121, 2, 11))
        # l = []
        # # use board.mirror(vertical=True), board.mirror(vertical=False), np.rot90
        # # to get the symmetries
        # for i, j in itertools.product(range(2), range(4)):
        #     newB = hn_utils.get_observation(board.mirror(vertical=i), player=board.turn)
        #     newB = np.rot90(newB, k=j)
        #     newPi = np.rot90(pi_board, k=j)
        #     newPi = np.reshape(newPi, 2662)
        #     l += [(newB, newPi)]
        # return l
        return [(hn_utils.get_observation(board, player=board.turn), pi)]
                

    @array_to_board
    def stringRepresentation(self, board: hn.BoardT) -> str:
        """Get a string representation of the board.

        Args:
            board: current board

        Returns:
            boardString: a string representation of the board."""
        return board.board_code()



if __name__ == "__main__":
    # test the getValidMoves function in a random game
    game = HnefataflGame()
    board = game.getInitBoard()

    while not game.getGameEnded(board, 1):
        valids = game.getValidMoves(board, 1)
        # make sure all the valid moves are valid
        for action in np.where(valids.flatten() == 1)[0]:
            move = hn_utils.action_to_move(action)
            # TODO: Fix getValidMoves returning invalid moves for some reason
            assert hn_utils.get_board(board).is_legal(move) if move is not None else True, (
                f"Move {move} ({action}) is not a legal move in board\n\n{board.unicode()}"
            )
        # choose a random move
        action = np.random.choice(np.where(valids.flatten() == 1)[0])
        board, _ = game.getNextState(board, 1, action)

    print("Game ended. Result:", game.getGameEnded(board, 1))

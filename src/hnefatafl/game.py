"""Game module for the Hnefatafl game. This module contains the :class:`Game`
class, which is responsible for managing the game and the players."""
from abc import ABCMeta, abstractmethod
import random
from token import COLONEQUAL
from hnefatafl import Board, BoardT, KingEscapeAndCaptureEasierBoard, Move, BLACK, WHITE, COLOR_NAMES, Color


# abstract base class
class Player(metaclass=ABCMeta):
    """Player class for playing Hnefatafl."""
    def __init__(self, color: Color) -> None:
        self.color = color

    @abstractmethod
    def get_move(self, board: BoardT) -> Move:
        """Get a move for the given board."""
        raise NotImplementedError
    

class HumanPlayer(Player):
    def __init__(self, color: Color, outputfn = str) -> None:
        super().__init__(color)
        self.p = outputfn

    def get_move(self, board: BoardT) -> Move:
        """Get a move for the given board."""
        # Print the board.
        print(self.p(board))
        # Get the move in pyhnefatafl notation from the user.
        move = input("Enter a move: ")
        # Parse the move.
        move = Move.from_code(move)
        # Return the move.
        return move
    

class RandomPlayer(Player):
    """Player that plays random legal moves."""
    def __init__(self, color: Color) -> None:
        super().__init__(color)

    def get_move(self, board: BoardT) -> Move:
        """Get a move for the given board."""
        move = random.choice(list(board.legal_moves))
        return move
    

class GreedyPlayer(Player):
    """Player that always captures if possible."""
    def __init__(self, color: Color) -> None:
        super().__init__(color)

    def get_move(self, board: BoardT) -> Move:
        """Get a move for the given board."""
        caps = board._all_capture_moves(board.turn)
        if caps:
            return random.choice(list(caps))
        else:
            return random.choice(list(board.legal_moves))
        
class GreedyWithKingPlayer(GreedyPlayer):
    """Player that always captures if possible, including the king."""
    def __init__(self, color: Color) -> None:
        super().__init__(color)

    def get_move(self, board: BoardT) -> Move:
        """Get a move for the given board."""
        # Check if the king can be captured.
        # If so, capture the king.
        # Otherwise, capture a piece.
        # If no pieces can be captured, move a piece.
        # try each move, if it captures the king, return it
        if self.color == BLACK:
            for move in board.legal_moves:
                if self._is_king_capture_move(move, board):
                    return move
        else: # white, king escapes
            for move in board.legal_moves:
                if self._is_king_escape_move(move, board):
                    return move
        caps = board._all_capture_moves(board.turn)
        if caps:
            return random.choice(list(caps))
        else:
            return random.choice(list(board.legal_moves))
        
    def _is_king_capture_move(self, move: Move, board: BoardT) -> bool:
        """Roll the board state forward and check if the king is captured."""
        # Push the move.
        board.push(move)
        # Check if the king is captured.
        is_king_capture = board.king_captured()
        # Pop the move.
        board.pop()
        # Return the result.
        return is_king_capture
    
    def _is_king_escape_move(self, move: Move, board: BoardT) -> bool:
        """Roll the board state forward and check if the king escapes."""
        # Push the move.
        board.push(move)
        # Check if the king escapes.
        is_king_escape = board.king_escaped()
        # Pop the move.
        board.pop()
        # Return the result.
        return is_king_escape
    
class Game:
    """Game class for running a game of Hnefatafl."""

    def __init__(self, board: BoardT, player_black: Player, player_white: Player, outputfn = str) -> None:
        self.board: BoardT = board()
        self.players = [player_black, player_white]
        self.p = outputfn

    def play(self) -> None:
        """Play a game of Hnefatafl."""
        while not self.board.is_game_over():
            # Get the current player's move.
            move = self.players[self.board.turn].get_move(self.board)
            # Play the move.
            self.board.push(move)
            # Output the move.
            print(f"Player {COLOR_NAMES[not self.board.turn]} played {move}.")
        # Print the board.
        print(self.output(self.board))
        # Print the winner.
        print(f"Winner: {COLOR_NAMES[self.board.winner]}")
        print(f"Win reason: {self.board.outcome().termination}")

    def output(self, board: BoardT) -> str:
        """Output the board."""
        return self.p(board)


if __name__ == "__main__":
    board = KingEscapeAndCaptureEasierBoard
    board.move_limit = 1000

    black = GreedyWithKingPlayer(BLACK)
    white = GreedyWithKingPlayer(WHITE)

    game = Game(board, player_black=black, player_white=white)
    # Play the game.
    game.play()

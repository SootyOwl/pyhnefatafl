"""Refactor of the classes in board.py."""
import itertools
from dataclasses import dataclass
import os
from typing import List, Tuple
import numpy as np
from enum import Enum


class PieceType(Enum):
    """Enum for the type of piece."""

    KING = "K"
    DEFENDER = "D"
    ATTACKER = "A"

    @property
    def is_defender(self) -> bool:
        """Return True if the piece is a defender."""
        return self in [PieceType.DEFENDER, PieceType.KING]

    @property
    def is_attacker(self) -> bool:
        """Return True if the piece is an attacker."""
        return self == PieceType.ATTACKER

    def is_enemy(self, other: "PieceType") -> bool:
        """Return True if the piece is an enemy of the other piece."""
        return (
            self.is_attacker
            and other.is_defender
            or self.is_defender
            and other.is_attacker
        )


# class so we can specify the position as a string, e.g. "A1" or "K5" rather than a tuple, but still use the tuple for calculations
class Position(tuple):
    """Class for a position on the board."""

    def __new__(cls, position):
        """Initialize the position. Accepts a tuple or a string.

        If a string is passed, it must be in the format "A1" or "K5", up to "K99", where the first character is the column and the rest is the row.
        """
        if isinstance(position, str):
            position = position.upper()
            # get the column from the first character
            column = ord(position[0]) - 65
            # get the row from the rest of the string
            row = int(position[1:]) - 1
            position = (row, column)
        return super().__new__(cls, position)

    def __str__(self):
        """Return the position as a string."""
        return chr(self[1] + 65) + str(self[0] + 1)

    def __repr__(self):
        """Return the position as a string."""
        return self.__str__()
    
    def __eq__(self, other):
        """Allow comparisons between Position and tuple, and Position and str."""
        if isinstance(other, tuple):
            return super().__eq__(other)
        elif isinstance(other, str):
            return self.__str__() == other
        else:
            return NotImplementedError


@dataclass
class Piece:
    """Class for a piece on the board."""

    position: Position
    piece_type: PieceType

    def move_to(self, new_position):
        """Move the piece to a new position."""
        self.position = new_position


@dataclass
class Board:
    """Class for the board."""

    state = np.zeros((11, 11))

    def __str__(self) -> str:
        # add a row of letters A-K to the top of the board
        board_str = (
            "  "
            + "   ".join([chr(i) for i in range(65, 65 + self.state.shape[1])])
            + "  \n"
        )
        board_str += "+---" * self.state.shape[1] + "+\n"
        for i, row in enumerate(self.state):
            # add a row number to the right of the board
            board_str += "| " + " | ".join(row) + " | " + str(i + 1) + "\n"
            board_str += "+---" * self.state.shape[1] + "+\n"
        return board_str

    def initialize_board(self):
        """Initialize the board."""
        self.state = np.array(
            [  #  A    B    C    D    E    F    G    H    I    J    K
                [" ", " ", " ", "A", "A", "A", "A", "A", " ", " ", " "],  # 1
                [" ", " ", " ", " ", " ", "A", " ", " ", " ", " ", " "],  # 2
                [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 3
                ["A", " ", " ", " ", " ", "D", " ", " ", " ", " ", "A"],  # 4
                ["A", " ", " ", " ", "D", "D", "D", " ", " ", " ", "A"],  # 5
                ["A", "A", " ", "D", "D", "K", "D", "D", " ", "A", "A"],  # 6
                ["A", " ", " ", " ", "D", "D", "D", " ", " ", " ", "A"],  # 7
                ["A", " ", " ", " ", " ", "D", " ", " ", " ", " ", "A"],  # 8
                [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],  # 9
                [" ", " ", " ", " ", " ", "A", " ", " ", " ", " ", " "],  # 10
                [" ", " ", " ", "A", "A", "A", "A", "A", " ", " ", " "],  # 11
            ]
        )

    def get_piece(self, position: Position) -> Piece:
        """Get the piece at a position."""
        if not isinstance(position, Position):
            position = Position(position)
        if self.state[position] == "K":
            return Piece(position, PieceType.KING)
        elif self.state[position] == "D":
            return Piece(position, PieceType.DEFENDER)
        elif self.state[position] == "A":
            return Piece(position, PieceType.ATTACKER)
        else:
            return None

    def get_pieces(self, piece_type: PieceType) -> List[Piece]:
        """Get all pieces of a certain type."""
        positions = np.where(self.state == piece_type.value)
        return [
            Piece(Position((row, column)), piece_type)
            for row, column in zip(*positions)
        ]

    def get_king_location(self) -> tuple:
        """Get the location of the king."""
        return np.where(self.state == "K")

    def get_possible_moves(self, piece: Piece) -> List[tuple]:
        """Get all possible moves for a piece. Returns a list of positions.

        Pieces move horizontally or vertically as far as they want, but are blocked by other pieces.
        All pieces move identically.
        """
        possible_moves = []
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, 11):
                new_position = (
                    piece.position[0] + i * direction[0],
                    piece.position[1] + i * direction[1],
                )
                if piece.piece_type != PieceType.KING and new_position in [
                    (5, 5),
                    (0, 0),
                    (0, 10),
                    (10, 0),
                    (10, 10),
                ]:
                    continue
                # if the new position is off the board, break
                if (
                    new_position[0] < 0
                    or new_position[0] > 10
                    or new_position[1] < 0
                    or new_position[1] > 10
                ):
                    break
                if self.state[new_position] != " ":
                    break
                possible_moves.append(new_position)
        # turn the list of tuples into a list of Positions
        possible_moves = [Position(position) for position in possible_moves]
        return possible_moves

    def move_piece(self, piece: Piece, new_position: Position):
        """Move a piece to a new position."""
        if new_position not in self.get_possible_moves(piece):
            raise ValueError("Invalid move.")
        self.state[piece.position] = " "
        self.state[new_position] = piece.piece_type.value
        piece.move_to(new_position)
        self.check_for_captures(piece)

    def get_adjacent_pieces(self, piece: Piece) -> List[Piece]:
        """Get all pieces adjacent to a piece."""
        adjacent_pieces = []
        for i, j in itertools.product(range(-1, 2), range(-1, 2)):
            if i == 0 and j == 0:
                continue
            new_position = (piece.position[0] + i, piece.position[1] + j)
            # if the new position is off the board, continue
            if (
                new_position[0] < 0
                or new_position[0] > 10
                or new_position[1] < 0
                or new_position[1] > 10
            ):
                continue
            if self.state[new_position] != " ":
                adjacent_pieces.append(self.get_piece(new_position))
        return adjacent_pieces

    def check_for_captures(self, moved_piece: Piece):
        """Check if any pieces are now sandwiched by this piece and an opponent."""
        for piece in self.get_adjacent_pieces(moved_piece):
            if piece.piece_type.is_enemy(moved_piece.piece_type):
                opp_pos = Position(
                    (
                        2 * piece.position[0] - moved_piece.position[0],
                        2 * piece.position[1] - moved_piece.position[1],
                    )
                )
                # check if opp_pos is on the board
                if opp_pos not in itertools.product(range(11), range(11)):
                    continue
                if self.state[opp_pos] == moved_piece.piece_type.value:
                    self.capture_piece(piece)

    def capture_piece(self, piece: Piece):
        """Capture a piece."""
        self.state[piece.position] = " "

    def check_king_surrounded(self) -> bool:
        """Check if the king is surrounded by attackers on all 4 sides, or if the king has attackers on 3 sides and the center tile on the 4th side."""
        king_location = self.get_king_location()
        adjacent_pieces = self.get_adjacent_pieces(self.get_piece(king_location))
        if len(adjacent_pieces) == 4 and all(
            piece.piece_type == PieceType.ATTACKER for piece in adjacent_pieces
        ):
            return True
        return (
            king_location in [(4, 5), (5, 4), (5, 6), (6, 5)]
            and sum(piece.piece_type == PieceType.ATTACKER for piece in adjacent_pieces)
            == 3
        )

    def check_king_escaped(self) -> bool:
        """Check if the king has escaped to the corner."""
        king_location = self.get_king_location()
        return king_location in [(0, 0), (0, 10), (10, 0), (10, 10)]
    
    # make the board subscriptable. board["A1"] returns the piece at that position
    def __getitem__(self, position: Position) -> Piece:
        return self.get_piece(position)


class AIPlayer:
    """Class for the AI player."""

    def __init__(self, network):
        """Initialize the AI player."""
        self.net = network

    def get_move(self, board) -> tuple:
        """Get the next move from the neural network."""
        ...


class HumanPlayer:
    """Class for the human player."""

    def __init__(self, player_id):
        """Initialize the human player."""
        self.player_id = player_id

    def get_move(self, board) -> Tuple[Position, Position]:
        """Get the next move from the user."""
        # clear the screen and print the board
        os.system("cls" if os.name == "nt" else "clear")
        print(board)
        print(f"Player {self.player_id} turn.")
        # get the piece to move
        piece = input("Enter the piece to move: ")
        # get the new position
        new_position = input("Enter the new position: ")
        if piece is None or new_position is None:
            raise ValueError("Invalid input.")
        return Position(piece), Position(new_position)


@dataclass
class Game:
    """Class for the game state.
    Contains methods for checking if the game is over, and for getting the winner."""

    board: Board

    def __init__(self):
        """Initialize the game."""
        self.board = Board()
        self.board.initialize_board()

    def is_over(self) -> bool:
        """Check if the game is over."""
        return self.board.check_king_surrounded() or self.board.check_king_escaped()

    def get_winner(self) -> PieceType:
        """Get the winner of the game."""
        if self.board.check_king_escaped():
            return PieceType.DEFENDER
        return PieceType.ATTACKER


class Play:
    """Class for playing the game."""

    def __init__(self, player_1, player_2):
        """Initialize the game."""
        self.game = Game()
        self.attacker = player_1
        self.defender = player_2
        self.current_player = self.attacker

    def play(self):
        """Play the game."""
        while True:
            player = self.current_player
            piece_pos, new_pos = player.get_move(self.game.board)
            piece = self.game.board.get_piece(piece_pos)
            if player == self.attacker and piece.piece_type.is_defender:
                print("You can only move attacker pieces!")
                continue
            if player == self.defender and piece.piece_type.is_attacker:
                print("You can only move defender pieces (or the king)!\n")
                continue
            try:
                self.game.board.move_piece(piece, new_pos)
            except ValueError:
                print("Invalid move. You can only move to empty spaces.")
                continue
            if self.game.is_over():
                break
            self.current_player = (
                self.attacker if self.current_player == self.defender else self.defender
            )
        print(f"The winner is {self.game.get_winner()}.")
        print(self.game.board)

    def execute_move_list(self, move_list: List[Tuple[Position, Position]]):
        """Execute a list of moves."""
        for piece_pos, new_pos in move_list:
            piece = self.game.board.get_piece(piece_pos)
            self.game.board.move_piece(piece, new_pos)
            if self.game.is_over():
                break
            self.current_player = (
                self.attacker if self.current_player == self.defender else self.defender
            )

        if self.game.is_over():
            print(f"The winner is {self.game.get_winner()}.")
            print(self.game.board)

    def get_fitness(self, player_id) -> float:
        """Get the fitness of the current player's position."""
        # if no player is provided, who is the current player?
        if player_id is None:
            player = self.current_player
        else:
            player = self.attacker if player_id == self.attacker.player_id else self.defender

        attacker: bool = player == self.attacker
        if self.game.is_over():
            if (
                attacker
                and self.game.get_winner() == PieceType.ATTACKER
            ):
                return 1
            if (
                not attacker
                and self.game.get_winner() == PieceType.DEFENDER
            ):
                return 1
            return 0
        # need a way to evaluate the fitness of the current position
        # try to use the number of moves to the corner as a fitness function
        # the fewer moves, the better
        # also include a penalty for each piece that is lost
        # the fewer pieces lost, the better
        # begin:
        king_location = self.game.board.get_king_location()
        # get the distance to the corner (the king starts in the middle of the 11x11 board)
        # the closer to the corner, the better
        distance_to_corner = min(
            king_location[0],
            king_location[1],
            10 - king_location[0],
            10 - king_location[1],
        )
        # get the number of pieces lost by the current player in proportion to the total number of pieces they had at the start
        # attackers start with 24 pieces, defenders start with 12
        pieces_lost = (
            (24 if attacker else 12)
            - len(self.game.board.get_pieces(PieceType.ATTACKER if attacker else PieceType.DEFENDER))
        )
        return distance_to_corner - pieces_lost


if __name__ == "__main__":
    play = Play(HumanPlayer("Attacker"), HumanPlayer("Defender"))
    # create a list of moves that will result in a win for the defender
    move_list = [
        ("D1", "D5"),
        ("F4", "B4"),
        ("D5", "D1"),
        ("F5", "F3"),  # king is unblocked
        ("D1", "D5"),
        ("F6", "F4"),  # move the king out of the center
        ("D5", "D1"),
        ("F4", "J4"),
        ("D1", "D5"),
        ("J4", "J1"),
        ("D5", "D1"),
    ]
    # convert the move list to a list of tuples of Position objects
    move_list = [(Position(piece), Position(new_pos)) for piece, new_pos in move_list]
    play.execute_move_list(move_list)
    print(play.get_fitness(HumanPlayer("Defender")))
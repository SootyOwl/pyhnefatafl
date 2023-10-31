"""Hnefatafl game engine. Implementation inspired by python-chess, but without
the bitboard optimizations.

Hnefatafl is a strategy board game for two players. The game is played on a
11x11 board with 12 men for white, a white king, and 24 men for black.
The goal of the game is for black to capture the king by surrounding it with pieces,
or for white to escape the king by moving it to the corner of the board.

The game is played by moving pieces orthogonally to a square on the same rank or file.
All pieces including the king move like a rook in chess. No piece may jump over another
piece, or move into an occupied square. Pieces are captured by sandwiching them between
two opposing pieces on the same rank or file, where the sandwich was created by a move
of an opposing piece. The king is captured by surrounding it with pieces on all four
sides.

The center of the board (F6) is a special square called the throne. This is the king's
starting position. No man may end its move on the throne. The king may end its move on
the throne. The throne is hostile to all pieces, and may be used to capture pieces by
sandwiching between an opposing piece and the throne. The king may not be captured by
sandwiching, as the king must be surrounded on all four sides to be captured, however
the king may be captured against the throne by three pieces (since the throne is
considered hostile to the king, this is a surrounded victory for black)."""

from collections import Counter
import copy
import dataclasses
import enum
from functools import partial
from itertools import chain
from time import sleep
import typing
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    SupportsInt,
    Tuple,
    Optional,
    Type,
    TypeVar,
    Union,
)

Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
PIECE_TYPES = [MAN, KING] = range(2)
PIECE_SYMBOLS = ["m", "k"]
PIECE_NAMES = ["man", "king"]


def piece_symbol(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_SYMBOLS[piece_type])


def piece_name(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_NAMES[piece_type])


UNICODE_PIECE_SYMBOLS = {
    "K": "♔",
    "k": "♚",
    "M": "♙",
    "m": "♟",
}
"""Unicode symbols for pieces."""

FILE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "#"]

STARTING_POSITION_CODE = (
    "3mmmmm3/5m5/11/m4M4m/m3MMM3m/mm1MMKMM1mm/m3MMM3m/m4M4m/11/5m5/3mmmmm3 b 0"
)
"""The starting position of the board, as a string (similar to FEN).
The first part is the board, the second part is the active color, and the third
part is the number of half-moves since the game started."""


class Status(enum.IntFlag):
    VALID = 0
    NO_KING = 1 << 0
    TOO_MANY_KINGS = 1 << 1
    BLACK_HAS_KING = 1 << 2
    TOO_MANY_WHITE_MEN = 1 << 3
    TOO_MANY_BLACK_MEN = 1 << 4


class Termination(enum.Enum):
    """Enum with reasons for the game to be over."""

    KING_ESCAPED = enum.auto()
    KING_CAPTURED = enum.auto()
    STALEMATE = enum.auto()

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")

    def __repr__(self) -> str:
        return f"Termination.{self.name}"


@dataclasses.dataclass
class Outcome:
    """Information about the outcome of an ended game, usually obtained from
    :func:`hnefatafl.Board.outcome()`.
    """

    termination: Termination
    """The reason for the game to be over."""

    winner: Optional[Color]
    """The winner of the game, or ``None`` if the game ended in a draw."""

    def result(self) -> str:
        """Return ``1-0`` if white won, ``0-1`` if black won, or ``1/2-1/2`` if
        the game ended in a draw."""
        return "1/2-1/2" if self.winner is None else ("1-0" if self.winner else "0-1")


class InvalidMoveError(ValueError):
    """Raised when move notation is not syntactically valid"""


class IllegalMoveError(ValueError):
    """Raised when the attempted move is illegal in the current position"""


Square = int
# fmt: off
SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1, I1, J1, K1,
    A2, B2, C2, D2, E2, F2, G2, H2, I2, J2, K2,
    A3, B3, C3, D3, E3, F3, G3, H3, I3, J3, K3,
    A4, B4, C4, D4, E4, F4, G4, H4, I4, J4, K4,
    A5, B5, C5, D5, E5, F5, G5, H5, I5, J5, K5,
    A6, B6, C6, D6, E6, F6, G6, H6, I6, J6, K6,
    A7, B7, C7, D7, E7, F7, G7, H7, I7, J7, K7,
    A8, B8, C8, D8, E8, F8, G8, H8, I8, J8, K8,
    A9, B9, C9, D9, E9, F9, G9, H9, I9, J9, K9,
    A10, B10, C10, D10, E10, F10, G10, H10, I10, J10, K10,
    A11, B11, C11, D11, E11, F11, G11, H11, I11, J11, K11,
] = range(121)
"""List of all squares on the board, in algebraic notation order."""
# fmt: on
SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]
"""List of all square names, in algebraic notation order."""


def parse_square(name: str) -> Square:
    """
    Gets the square index for the given square *name*.
    (e.g., ``a1`` returns ``0``, ``b2`` returns ``12``, etc.)

    :raises: :exc:`ValueError` if the square name is invalid
    """
    try:
        return SQUARE_NAMES.index(name.upper())
    except ValueError:
        raise ValueError(f"Invalid square name: {name!r}")

def square_name(square: Square) -> str:
    """
    Gets the square name for the given square *index*.
    (e.g., ``0`` returns ``a1``, ``12`` returns ``b2``, etc.)
    """
    return SQUARE_NAMES[square]


def square(file_index: int, rank_index: int) -> Square:
    """
    Gets the square index for the given file and rank indices.
    (e.g., ``0, 0`` returns ``0``, ``1, 1`` returns ``12``, etc.)
    """
    return rank_index * 11 + file_index


def square_file(square: Square) -> int:
    """
    Gets the file index for the given square where ``0`` is the a-file.
    """
    return square % 11


def square_rank(square: Square) -> int:
    """
    Gets the rank index for the given square where ``0`` is the 1st rank.
    """
    return square // 11


def square_distance(a: Square, b: Square) -> int:
    """
    Gets the distance (i.e., the number of orthogonal, non-diagonal, single-moves) from square *a* to square *b*.
    Essentially, this is the taxicab distance.
    """
    return abs(square_file(a) - square_file(b)) + abs(square_rank(a) - square_rank(b))


def square_mirror(sq: Square, vertical=True) -> Square:
    """
    Gets the mirror image of the given square.

    >>> square_mirror(A1)
    A11
    >>> square_mirror(A11)
    A1
    >>> square_mirror(B2)
    B10
    >>> square_mirror(B10)
    B2
    >>> square_mirror(C3)
    C9

    >>> square_mirror(A1, vertical=False)
    K1
    >>> square_mirror(A11, vertical=False)
    K11
    >>> square_mirror(B2, vertical=False)
    J2
    >>> square_mirror(B10, vertical=False)
    J10
    """
    if vertical:
        return square(square_file(sq), 10 - square_rank(sq))
    else:
        return square(10 - square_file(sq), square_rank(sq))


# try a simple bitboard implementation, represent the 11x11 board as a 121-bit integer
Bitboard = int
BB_EMPTY = 0
BB_ALL = (1 << 121) - 1
# fmt: off
BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1, BB_I1, BB_J1, BB_K1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2, BB_I2, BB_J2, BB_K2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3, BB_I3, BB_J3, BB_K3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4, BB_I4, BB_J4, BB_K4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5, BB_I5, BB_J5, BB_K5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6, BB_I6, BB_J6, BB_K6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7, BB_I7, BB_J7, BB_K7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8, BB_I8, BB_J8, BB_K8,
    BB_A9, BB_B9, BB_C9, BB_D9, BB_E9, BB_F9, BB_G9, BB_H9, BB_I9, BB_J9, BB_K9,
    BB_A10, BB_B10, BB_C10, BB_D10, BB_E10, BB_F10, BB_G10, BB_H10, BB_I10, BB_J10, BB_K10,
    BB_A11, BB_B11, BB_C11, BB_D11, BB_E11, BB_F11, BB_G11, BB_H11, BB_I11, BB_J11, BB_K11,
] = [1 << sq for sq in SQUARES]

BB_CORNERS = BB_A1 | BB_K1 | BB_A11 | BB_K11
BB_CENTER = BB_KING_START = BB_THRONE = BB_F6

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H,
    BB_FILE_I,
    BB_FILE_J,
    BB_FILE_K,
] = [0x4008_0100_2004_0080_1002_0040_0801 << i for i in range(11)]

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
    BB_RANK_9,
    BB_RANK_10,
    BB_RANK_11,
] = [0x7FF << (11 * i) for i in range(11)]

BB_EDGE = (
    (BB_RANK_1 | BB_RANK_11) |
    (BB_FILE_A | BB_FILE_K)
)

BB_ATTACKERS = (
    (BB_RANK_1 ^ (BB_A1 | BB_B1 | BB_C1 | BB_I1 | BB_J1 | BB_K1)) |
    (BB_FILE_A ^ (BB_A1 | BB_A2 | BB_A3 | BB_A9 | BB_A10 | BB_A11)) |
    (BB_RANK_11 ^ (BB_A11 | BB_B11 | BB_C11 | BB_I11 | BB_J11 | BB_K11)) |
    (BB_FILE_K ^ (BB_K1 | BB_K2 | BB_K3 | BB_K9 | BB_K10 | BB_K11)) |
    (BB_F2 | BB_J6 | BB_F10 | BB_B6)
)
"""Bitboard representing the starting position of black pieces."""

BB_DEFENDERS = (  # white's start has the king in the center, and the other 12 pieces surrounding it in a diamond shape
    BB_D6 | BB_E6 | BB_G6 | BB_H6 | BB_F4 | BB_F5 | BB_F7 | BB_F8 | BB_E5 | BB_E7 | BB_G5 | BB_G7
)
"""Bitboard representing the starting position of white pieces."""
# fmt: on


def lsb(bb: Bitboard) -> int:
    """
    Gets the index of the least significant bit in the given bitboard.
    This index is the same as the square index for the square corresponding to the bit.

    >>> lsb(BB_A1)
    0
    >>> lsb(BB_C11)

    >>> lsb(BB_B1 | BB_C1 | BB_D1)
    1
    """
    return (bb & -bb).bit_length() - 1


def scan_forward(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r


def msb(bb: Bitboard) -> int:
    return bb.bit_length() - 1


def scan_reversed(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = msb(bb)
        yield r
        bb ^= BB_SQUARES[r]


popcount: Callable[[Bitboard], int] = getattr(
    int, "bit_count", lambda bb: bin(bb).count("1")
)


def shift_down(b: Bitboard) -> Bitboard:
    return b >> 11


def shift_2_down(b: Bitboard) -> Bitboard:
    return b >> 22


def shift_up(b: Bitboard) -> Bitboard:
    return (b << 11) & BB_ALL


def shift_2_up(b: Bitboard) -> Bitboard:
    return (b << 22) & BB_ALL


def shift_right(b: Bitboard) -> Bitboard:
    return (b << 1) & ~BB_FILE_A & BB_ALL


def shift_2_right(b: Bitboard) -> Bitboard:
    return (b << 2) & ~BB_FILE_A & ~BB_FILE_B & BB_ALL


def shift_left(b: Bitboard) -> Bitboard:
    return (b >> 1) & ~BB_FILE_K


def shift_2_left(b: Bitboard) -> Bitboard:
    return (b >> 2) & ~BB_FILE_J & ~BB_FILE_K


def _carry_rippler(mask: Bitboard) -> Iterator[Bitboard]:
    """Yields all subsets of the given bitboard."""
    subset = BB_EMPTY
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break


def flip_vertical(mask: Bitboard) -> Bitboard:
    """Perform a vertical flip, or a little-endian to big-endian conversion."""
    return sum(((mask >> (11 * i)) & BB_RANK_1) << (11 * (10 - i)) for i in range(11))


def flip_horizontal(mask: Bitboard) -> Bitboard:
    """Perform a horizonal flip, or a bit reversal."""
    return sum(((mask >> i) & BB_FILE_A) << (10 - i) for i in range(11))


@dataclasses.dataclass
class Piece:
    """A piece with type and color."""

    piece_type: PieceType
    """The piece type."""

    color: Color
    """The piece color."""

    def symbol(self) -> str:
        """Returns the piece symbol ``M``, ``m``, or ``K``."""
        symbol = piece_symbol(self.piece_type)
        return symbol.upper() if self.color else symbol

    def unicode_symbol(self, *, invert_color: bool = False) -> str:
        symbol = self.symbol().swapcase() if invert_color else self.symbol()
        return UNICODE_PIECE_SYMBOLS[symbol]

    def __hash__(self) -> int:
        return self.piece_type + (-1 if self.color else 2)

    def __str__(self) -> str:
        return self.symbol()

    def __repr__(self) -> str:
        return f"Piece.from_symbol({self.symbol()!r})"

    @classmethod
    def from_symbol(cls, symbol: str) -> "Piece":
        """Creates a :class:`~hnefatafl.Piece` from a piece symbol.

        :raises: :exc:`ValueError` if the symbol is invalid
        """
        return cls(PIECE_SYMBOLS.index(symbol.lower()), symbol.isupper())


@dataclasses.dataclass(unsafe_hash=True)
class Move:
    """
    Represents a move from a square to another square.
    """

    from_square: Square
    """The source square."""

    to_square: Square
    """The target square."""

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"Move.from_code({self.code!r})"

    def __bool__(self) -> bool:
        return bool(self.from_square != self.to_square)

    @property
    def code(self) -> str:
        """Returns the move code in algebraic notation.

        For example, ``A4.B`` indicates the piece moves from A4 to B4, and
        ``A5.4`` indicates the piece moves from A4 to A5.
        As all pieces move orthogonally, the move code is simply the source
        square followed by the target square's rank OR file, depending on
        which is different.
        """
        # if the file is the same, the move is along a rank, if the rank is the same, the move is along a file, otherwise it's invalid
        if square_file(self.from_square) == square_file(self.to_square):
            return f"{square_name(self.from_square)}.{square_name(self.to_square)[1]}"
        elif square_rank(self.from_square) == square_rank(self.to_square):
            return f"{square_name(self.from_square)}.{square_name(self.to_square)[0]}"
        else:
            return f"{square_name(self.from_square)}.{square_name(self.to_square)}"

    @classmethod
    def from_code(cls, code: str) -> "Move":
        """Creates a :class:`~hnefatafl.Move` from a move code.

        :raises: :exc:`ValueError` if the move code is invalid.
        """
        if len(code) != 4:
            raise ValueError(f"Move code must be 4 characters, got {code!r}")

        # check if this is a move along a rank or file by looking at the last character
        first, last = code.split(".")
        try:
            parse_square(first)

            if last in RANK_NAMES:
                return cls(parse_square(first), parse_square(first[0] + last))
            elif last in FILE_NAMES:
                return cls(parse_square(first), parse_square(last + first[1]))
            else:
                raise ValueError(f"Invalid move code {code!r}")
        except ValueError as e:
            raise ValueError(f"Invalid move code {code!r}") from e

    @classmethod
    def null(cls) -> "Move":
        """
        Gets a null move.

        A null move just passes the turn to the other player. Null moves evaluate to ``False``.

        >>> import hnefatafl
        >>>
        >>> bool(hnefatafl.Move.null())
        False
        """
        return cls(0, 0)

    def is_null(self) -> bool:
        """
        Returns ``True`` if this is a null move.
        """
        return self.from_square == self.to_square


BaseBoardT = TypeVar("BaseBoardT", bound="BaseBoard")


class BaseBoard:
    """
    A board representing the position of hnafatafl pieces. See
    :class:`~hnefatafl.Board` for a full board with move generation.

    The board is initialised with the pieces in their starting positions,
    unless otherwise specified in the optional *board_code* argument. If
    *board_code* is ``None``, an empty board is created.
    """

    def __init__(self, board_code: Optional[str] = STARTING_POSITION_CODE) -> None:
        self.occupied_co = [BB_EMPTY, BB_EMPTY]

        if board_code is None:
            self._clear_board()
        elif board_code == STARTING_POSITION_CODE:
            self._reset_board()
        else:
            self._set_board_code(board_code)

    def _reset_board(self) -> None:
        self.kings = BB_THRONE
        self.men = BB_DEFENDERS | BB_ATTACKERS
        self.occupied_co[WHITE] = BB_DEFENDERS | BB_THRONE
        self.occupied_co[BLACK] = BB_ATTACKERS
        self.occupied = BB_DEFENDERS | BB_ATTACKERS | BB_THRONE

    def reset_board(self) -> None:
        """
        Resets the board to the starting position.

        :class:`~hnefatafl.Board` also resets the move stack, but not turn,
        or move counters. Use :func:`~hnefatafl.Board.reset()` to fully reset
        the starting position.
        """
        self._reset_board()

    def _clear_board(self) -> None:
        self.men = BB_EMPTY
        self.kings = BB_EMPTY
        self.occupied_co[WHITE] = BB_EMPTY
        self.occupied_co[BLACK] = BB_EMPTY
        self.occupied = BB_EMPTY

    def clear_board(self) -> None:
        """
        Clears the board of all pieces.

        :class:`~hnefatafl.Board` also resets the move stack."""
        self._clear_board()

    def pieces_mask(self, piece_type: PieceType, color: Color) -> Bitboard:
        """
        Returns a bitboard of the pieces of the given *piece_type* and *color*.
        """
        if piece_type == KING:
            bb = self.kings
        elif piece_type == MAN:
            bb = self.men
        else:
            assert False, f"expected PieceType, got {piece_type!r}"

        return bb & self.occupied_co[color]

    def pieces(self, piece_type: PieceType, color: Color) -> "SquareSet":
        """
        Gets pieces of the given type and color.
        """
        return SquareSet(self.pieces_mask(piece_type, color))

    def piece_at(self, square: Square) -> Optional[Piece]:
        """Gets the :class:`piece <hnefatafl.Piece>` at the given *square*."""
        piece_type = self.piece_type_at(square)
        if piece_type is None:
            return None
        mask = BB_SQUARES[square]
        color = bool(self.occupied_co[WHITE] & mask)
        return Piece(piece_type, color)

    def piece_type_at(self, square: Square) -> Optional[PieceType]:
        mask = BB_SQUARES[square]

        if not self.occupied & mask:
            return None  # no piece at square
        elif self.kings & mask:
            return KING
        else:
            return MAN

    def color_at(self, square: Square) -> Optional[Color]:
        """Gets the color at the given *square*."""
        mask = BB_SQUARES[square]
        if self.occupied_co[WHITE] & mask:
            return WHITE
        elif self.occupied_co[BLACK] & mask:
            return BLACK
        else:
            return None

    def is_occupied(self, square: Square) -> bool:
        """Checks if the given *square* is occupied."""
        return bool(self.occupied & BB_SQUARES[square])

    def moves(self, square: Square) -> "SquareSet":
        """Gets the set of squares that the piece at the given *square* can move to."""
        return SquareSet(self._moves(square))

    def _moves(self, square: Square) -> Bitboard:
        piece = self.piece_type_at(square)
        if piece is None:
            return BB_EMPTY  # no piece at square

        occupied = self.occupied
        bb = BB_SQUARES[square]
        up = shift_up(bb) & ~occupied
        down = shift_down(bb) & ~occupied
        left = shift_left(bb) & ~occupied
        right = shift_right(bb) & ~occupied

        up_moves = up
        while up_moves:
            up_moves = shift_up(up_moves) & ~occupied
            up |= up_moves

        down_moves = down
        while down_moves:
            down_moves = shift_down(down_moves) & ~occupied
            down |= down_moves

        left_moves = left
        while left_moves:
            left_moves = shift_left(left_moves) & ~occupied
            left |= left_moves

        right_moves = right
        while right_moves:
            right_moves = shift_right(right_moves) & ~occupied
            right |= right_moves

        valid_moves = up | down | left | right

        if piece != KING:
            # remove BB_CORNERS and BB_THRONE from valid moves
            valid_moves &= ~(BB_CORNERS | BB_THRONE)
        return valid_moves

    def _captures_possible(
        self, square: Square
    ) -> Tuple[Bitboard, Bitboard, Dict[Square, List[Square]]]:
        piece = self.piece_type_at(square)
        if piece is None:
            return BB_EMPTY, BB_EMPTY, {}  # no piece at square

        bb = BB_SQUARES[square]
        moves = self._moves(square)

        player = (
            self.occupied_co[WHITE]
            if self.color_at(square) == WHITE
            else self.occupied_co[BLACK]
        )
        opponent = (
            self.occupied_co[BLACK]
            if self.color_at(square) == WHITE
            else self.occupied_co[WHITE]
        )

        captures = BB_EMPTY
        pieces_captured_by_move = BB_EMPTY
        capture_dict = {}
        for move in scan_forward(moves):
            capture_dict[move] = 0
            # we need to look at each move and check two steps in each direction
            # if we find an enemy piece adjacent and a friendly piece on the other side,
            # this move is a capture
            # the king cannot be captured, so exclude it from the check
            # check up
            up = shift_up(BB_SQUARES[move]) & opponent & ~self.kings
            if up:
                # check if there is a friendly piece or the throne or a corner two squares up
                # if the king is on the throne, the throne check is not valid
                if self.kings & BB_THRONE:
                    up = shift_up(up) & (player | BB_CORNERS)
                else:
                    up = shift_up(up) & (player | BB_THRONE | BB_CORNERS)
            if up:
                captures |= BB_SQUARES[move]
                pieces_captured_by_move |= shift_up(BB_SQUARES[move])
                capture_dict[move] |= shift_up(BB_SQUARES[move])
            # check down
            down = shift_down(BB_SQUARES[move]) & opponent & ~self.kings
            if down:
                if self.kings & BB_THRONE:
                    down = shift_down(down) & (player | BB_CORNERS)
                else:
                    down = shift_down(down) & (player | BB_THRONE | BB_CORNERS)
            if down:
                captures |= BB_SQUARES[move]
                pieces_captured_by_move |= shift_down(BB_SQUARES[move])
                capture_dict[move] |= shift_down(BB_SQUARES[move])
            # check left
            left = shift_left(BB_SQUARES[move]) & opponent & ~self.kings
            if left:
                if self.kings & BB_THRONE:
                    left = shift_left(left) & (player | BB_CORNERS)
                else:
                    left = shift_left(left) & (player | BB_THRONE | BB_CORNERS)
            if left:
                captures |= BB_SQUARES[move]
                pieces_captured_by_move |= shift_left(BB_SQUARES[move])
                capture_dict[move] |= shift_left(BB_SQUARES[move])
            # check right
            right = shift_right(BB_SQUARES[move]) & opponent & ~self.kings
            if right:
                if self.kings & BB_THRONE:
                    right = shift_right(right) & (player | BB_CORNERS)
                else:
                    right = shift_right(right) & (player | BB_THRONE | BB_CORNERS)
            if right:
                captures |= BB_SQUARES[move]
                pieces_captured_by_move |= shift_right(BB_SQUARES[move])
                capture_dict[move] |= shift_right(BB_SQUARES[move])

        return captures, pieces_captured_by_move, capture_dict

    def captures(self, square: Square) -> "SquareSet":
        """Gets the set of squares that the piece at the given *square* can capture."""
        return SquareSet(self._captures(square))

    def _captures(self, square: Square) -> Bitboard:
        captures, _, _ = self._captures_possible(square)
        return captures

    def _threatened_pieces(self) -> Bitboard:
        threatened = BB_EMPTY
        for square in scan_forward(self.occupied):
            threatened |= self._captures(square)
        return threatened

    def _threatened_by(self, square: Square) -> Bitboard:
        threatened = BB_EMPTY
        for move in scan_forward(self._moves(square)):
            threatened |= self._captures(move)
        return threatened
    
    # compute all possible captures at the current board state
    def _all_capture_moves(self, color: Color) -> List[Move]:
        moves = []
        for square in scan_forward(self.occupied_co[color]):
            captures, pieces_captured_by_move, capture_dict = self._captures_possible(square)
            if captures:
                for move in scan_forward(captures):
                    moves.append(Move(square, move))
        return moves

    def _get_adjacent_squares(self, square: Square) -> Bitboard:
        bb = BB_SQUARES[square]
        up = shift_up(bb)
        down = shift_down(bb)
        left = shift_left(bb)
        right = shift_right(bb)
        return up | down | left | right

    def _get_adjacent_pieces(self, square: Square) -> Bitboard:
        return self._get_adjacent_squares(square) & self.occupied

    def _get_adjacent_opponent_pieces(self, square: Square, color: Color) -> Bitboard:
        return self._get_adjacent_squares(square) & self.occupied_co[not color]

    def _get_adjacent_friendly_pieces(self, square: Square, color: Color) -> Bitboard:
        return self._get_adjacent_squares(square) & self.occupied_co[color]

    def get_adjacent_squares(self, square: Square) -> "SquareSet":
        """Gets the set of squares adjacent to the given *square*."""
        return SquareSet(self._get_adjacent_squares(square))

    def get_adjacent_pieces(self, square: Square) -> "SquareSet":
        """Gets the set of pieces adjacent to the given *square*."""
        return SquareSet(self._get_adjacent_pieces(square))

    def get_adjacent_opponent_pieces(self, square: Square, color: Color) -> "SquareSet":
        """Gets the set of opponent pieces adjacent to the given *square*."""
        return SquareSet(self._get_adjacent_opponent_pieces(square, color))

    def get_adjacent_friendly_pieces(self, square: Square, color: Color) -> "SquareSet":
        """Gets the set of friendly pieces adjacent to the given *square*."""
        return SquareSet(self._get_adjacent_friendly_pieces(square, color))

    def _get_adjacent_empty_squares(self, square: Square) -> Bitboard:
        return self._get_adjacent_squares(square) & ~self.occupied

    def get_adjacent_empty_squares(self, square: Square) -> "SquareSet":
        """Gets the set of empty squares adjacent to the given *square*."""
        return SquareSet(self._get_adjacent_empty_squares(square))

    def _remove_piece_at(self, square: Square) -> Optional[PieceType]:
        """Removes the piece at the given *square* and returns its type.

        If the piece was a king, the board is left in an inconsistent state. Use
        :func:`~chess.Board.remove_piece_at()` instead."""
        piece_type = self.piece_type_at(square)
        mask = BB_SQUARES[square]

        if piece_type == MAN:
            self.men ^= mask
        elif piece_type == KING:
            self.kings ^= mask
        else:
            return

        self.occupied ^= mask
        self.occupied_co[WHITE] &= ~mask
        self.occupied_co[BLACK] &= ~mask

        return piece_type

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        """Removes the piece at the given *square* and returns the piece."""
        color = bool(self.occupied_co[WHITE] & BB_SQUARES[square])
        piece_type = self._remove_piece_at(square)
        return Piece(color, piece_type) if piece_type else None

    def _set_piece_at(
        self, square: Square, piece_type: PieceType, color: Color
    ) -> None:
        self._remove_piece_at(square)

        mask = BB_SQUARES[square]

        if piece_type == MAN:
            self.men |= mask
        elif piece_type == KING:
            self.kings |= mask
        else:
            return

        self.occupied ^= mask
        self.occupied_co[color] ^= mask

    def set_piece_at(self, square: Square, piece: Piece) -> None:
        """Sets the piece at the given *square*."""
        if piece is None:
            self._remove_piece_at(square)
        else:
            self._set_piece_at(square, piece.piece_type, piece.color)

    def board_code(self) -> str:
        """
        Gets the board code (similar to FEN) of the board.
        (e.g., "3mmmmm3/5m5/11/m4M4m/m3MMM3m/mm1MMKMM1mm/m3MMM3m/m4M4m/11/5m5/3mmmmm3")
        """
        builder = []
        empty = 0

        for square in SQUARES:
            if piece := self.piece_at(square):
                if empty:
                    builder.append(str(empty))
                    empty = 0

                builder.append(piece.symbol())

            else:
                empty += 1
            if BB_SQUARES[square] & BB_FILE_K:
                if empty:
                    builder.append(str(empty))
                    empty = 0

                if square != K11:
                    builder.append("/")

        return "".join(builder)

    def _set_board_code(self, code: str) -> None:
        code = code.strip()
        if " " in code:
            raise ValueError(
                f"expected position part of code, got multiple parts: {code!r}"
            )

        # ensure that the code is valid
        rows = code.split("/")
        if len(rows) != 11:
            raise ValueError(f"expected 11 rows, got {len(rows)}: {code!r}")

        # validate each row
        squares = [[] for _ in range(11)]
        for row_squares, row in zip(squares, rows):
            # scan through the row and get the squares
            # keep track of the previous character to ensure that we handle two-digit numbers correctly
            previous = None
            for c in row:
                # if the character is a digit, then we need to check the previous character
                if c.isdigit():
                    # if the previous character is also a digit, then we have a two-digit number
                    if previous and previous.isdigit():
                        row_squares.extend([None] * int(previous + c))
                        previous = None
                    # otherwise, we have a single-digit number, so we skip it and set the previous character to the digit
                    else:
                        previous = c
                        continue
                # if the character is not a digit, then we need to check if it is a valid piece character
                elif c.lower() not in PIECE_SYMBOLS:
                    raise ValueError(f"invalid character in board code: {c!r}")
                # if the character is a valid piece character, then we increment the number of squares by 1
                else:
                    # handle the previous character if it was a digit
                    if previous and previous.isdigit():
                        row_squares.extend([None] * int(previous))
                        previous = None
                    # add the piece to the row
                    row_squares.append(c)
            # if the previous character was a digit, then we need to handle it
            if previous and previous.isdigit():
                row_squares.extend([None] * int(previous))
            # if the number of squares is not 11, then the row is invalid
            if len(row_squares) != 11:
                raise ValueError(f"expected 11 squares in row, got {squares}: {row!r}")

        # set the board
        self._clear_board()
        for square, piece in zip(SQUARES, chain.from_iterable(squares)):
            if piece:
                self._set_piece_at(
                    square, PIECE_SYMBOLS.index(piece.lower()), piece.isupper()
                )

    def piece_map(self, *, mask: Bitboard = BB_ALL) -> Dict[Square, Piece]:
        """Gets a dictionary mapping squares to pieces."""
        return {
            square: self.piece_at(square)
            for square in scan_reversed(mask & self.occupied)
        }

    def _set_piece_map(self, pieces: Mapping[Square, Piece]) -> None:
        self._clear_board()
        for square, piece in pieces.items():
            self._set_piece_at(square, piece.piece_type, piece.color)

    def set_piece_map(self, pieces: Mapping[Square, Piece]) -> None:
        """Sets the board from a dictionary mapping squares to pieces."""
        self._set_piece_map(pieces)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.board_code()!r})"

    def __str__(self) -> str:
        builder = []
        for square in SQUARES:
            if piece := self.piece_at(square):
                builder.append(piece.symbol())
            else:
                builder.append(".")
            if BB_SQUARES[square] & BB_FILE_K and square != K11:
                builder.append("\n")

        builder = self._add_file_names(builder)
        builder = self._add_rank_names(builder)
        return "".join(builder)

    def _add_file_names(self, builder: List[str]) -> List[str]:
        """Adds the file (a-k) names to the top of the board string.

        The file names are added to the beginning of each file, in other words,
        the file names are added to the top of the board.
        """
        # add the file names to the top of the board
        file_string = "".join(FILE_NAMES)
        builder.insert(0, f"{file_string}\n")
        return builder

    def _add_rank_names(self, builder: List[str]) -> List[str]:
        """Adds the rank (1-11) names to the board string.

        The rank names are added to the end of each rank. 10 and 11 are
        represented by the symbols '+' and '#', respectively.
        Args:
            builder: The list of strings to add the rank names to."""
        # find the indexes of the newlines and insert the rank names before them
        newline_indexes = [i for i, c in enumerate(builder) if c == "\n"]
        for i, index in enumerate(newline_indexes):
            builder.insert(index + i, f" {RANK_NAMES[i]}")

        # add the last rank name
        builder.append(f" {RANK_NAMES[-1]}")
        return builder

    def unicode(
        self,
        *,
        invert_color: bool = False,
        borders: bool = True,
        empty_square: str = "⭘",
    ) -> str:
        """
        Gets a Unicode string representation of the board.
        """
        builder = []
        for rank_index in range(11):
            if borders:
                builder.extend(("  ", "-" * 23, "\n", RANK_NAMES[rank_index], " "))
            for file_index in range(11):
                square_index = square(file_index, rank_index)

                if borders:
                    builder.append("|")
                elif file_index > 0:
                    builder.append(" ")

                if piece := self.piece_at(square_index):
                    builder.append(piece.unicode_symbol(invert_color=invert_color))
                else:
                    builder.append(empty_square)

            if borders:
                builder.append("|")

            if borders or (rank_index < 10):
                builder.append("\n")

        if borders:
            builder.extend(("  ", "-" * 23, "\n", "   ", " ".join(FILE_NAMES), "\n"))
        return "".join(builder)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseBoard):
            return self.board_code() == other.board_code()
        return NotImplemented

    def apply_transform(self, f: Callable[[Bitboard], Bitboard]) -> None:
        self.men = f(self.men)
        self.kings = f(self.kings)

        self.occupied_co[WHITE] = f(self.occupied_co[WHITE])
        self.occupied_co[BLACK] = f(self.occupied_co[BLACK])
        self.occupied = f(self.occupied)

    def transform(self: BaseBoardT, f: Callable[[Bitboard], Bitboard]) -> BaseBoardT:
        board = copy.copy(self)
        board.apply_transform(f)
        return board

    @classmethod
    def empty(cls: Type[BaseBoardT]) -> BaseBoardT:
        return cls(None)

    def copy(self: BaseBoardT) -> BaseBoardT:
        return copy.copy(self)

    def mirror(self: BaseBoardT, vertical=True) -> BaseBoardT:
        """Mirrors the board."""
        return (
            self.transform(flip_vertical)
            if vertical
            else self.transform(flip_horizontal)
        )


BoardT = TypeVar("BoardT", bound="Board")


class _BoardState(Generic[BoardT]):
    def __init__(self, board: BoardT) -> None:
        self.men = board.men
        self.kings = board.kings

        self.occupied_w = board.occupied_co[WHITE]
        self.occupied_b = board.occupied_co[BLACK]
        self.occupied = board.occupied

        self.turn = board.turn
        self.halfmove_clock = board.halfmove_clock
        self.fullmove_number = board.fullmove_number

    def restore(self, board: BoardT) -> None:
        board.men = self.men
        board.kings = self.kings

        board.occupied_co[WHITE] = self.occupied_w
        board.occupied_co[BLACK] = self.occupied_b
        board.occupied = self.occupied

        board.turn = self.turn
        board.halfmove_clock = self.halfmove_clock
        board.fullmove_number = self.fullmove_number


class Board(BaseBoard):
    """
    A :class:`~chess.BaseBoard`, additional information representing
    a hnefatafl position, and a move stack.

    Provides endgame detection, the capability to make and unmake moves.

    Initialised to standard starting position unless otherwise specified with a
    hnefatafl board code.
    """

    starting_code: str = STARTING_POSITION_CODE
    turn: Color = BLACK
    """The side to move (``chess.WHITE`` or ``chess.BLACK``)."""

    move_limit: int = 100
    """The maximum number of moves allowed in a game."""

    def __init__(self: BoardT, code: Optional[str] = STARTING_POSITION_CODE) -> None:
        BaseBoard.__init__(self, None)
        self.turn = BLACK
        self.move_stack = []
        self._captured_pieces: Dict[Color, Piece] = {WHITE: [], BLACK: []}
        self._stack: List[_BoardState[BoardT]] = []
        self._last_move: List[Move] = [Move.null(), Move.null()]

        self.fullmove_number: int = 1
        """Counts the number of full moves. It starts at 1, and is incremented after White's move."""

        self.halfmove_clock: int = 0
        """The number of halfmoves since the last capture."""

        if code is None:
            self.clear()
        elif code == type(self).starting_code:
            self.reset()
        else:
            self.set_code(code)

    @property
    def legal_moves(self) -> "LegalMoveGenerator":
        """
        A dynamically generated list of legal moves.
        """
        return LegalMoveGenerator(self)

    def reset(self) -> None:
        """
        Resets the board to the starting position.
        """
        self.turn = BLACK
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.reset_board()

    def reset_board(self) -> None:
        super().reset_board()
        self.clear_stack()

    def clear(self) -> None:
        """Clears the board. Resets move stack and move counters."""
        self.turn = BLACK
        self.halfmove_clock = 0
        self.fullmove_number = 1

        self.clear_board()

    def clear_board(self) -> None:
        super().clear_board()
        self.clear_stack()

    def clear_stack(self) -> None:
        """Clears the move stack."""
        self.move_stack.clear()
        self._stack.clear()

    def set_code(self, code: str) -> None:
        """Sets the board from a board code."""
        # split the code into parts
        parts = code.strip().split(" ")
        # if there are no parts, then the code is invalid
        if not parts:
            raise ValueError(f"expected board code, got empty string: {code!r}")
        # the first part is the position part, which is required
        position = parts[0]
        # the second part is the turn part ('b' or 'w'), which is optional
        turn = parts[1] if len(parts) > 1 else None
        # the third part is the fullmove number, which is optional
        fullmove_number = parts[2] if len(parts) > 2 else None

        # set the position
        self._set_board_code(position)

        # set the turn
        if turn:
            if turn not in "bw":
                raise ValueError(f"invalid turn in board code: {turn!r}")
            self.turn = WHITE if turn == "w" else BLACK
            # set the halfmove clock
            self.halfmove_clock = 1 if self.turn is WHITE else 0
        # set the fullmove number
        if fullmove_number:
            try:
                self.fullmove_number = int(fullmove_number)
            except ValueError as e:
                raise ValueError(
                    f"invalid fullmove number in board code: {fullmove_number!r}"
                ) from e

            if self.fullmove_number < 1:
                raise ValueError(
                    f"fullmove number must be positive: {fullmove_number!r}"
                )

    def get_code(self) -> str:
        """Returns a board code."""
        return f"{self.board_code()} {self._get_turn_code()} {self._get_fullmove_number_code()}"

    def _get_turn_code(self) -> str:
        """Returns a turn code."""
        return "w" if self.turn is WHITE else "b"

    def _get_fullmove_number_code(self) -> str:
        """Returns a fullmove number code."""
        return str(self.fullmove_number)

    def ply(self) -> int:
        return 2 * (self.fullmove_number - 1) + (self.turn == WHITE)

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        piece = super().remove_piece_at(square)
        self.clear_stack()
        return piece

    def set_piece_at(self, square: Square, piece: Optional[Piece]) -> None:
        super().set_piece_at(square, piece)
        self.clear_stack()

    def generate_legal_moves(
        self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL
    ) -> Iterator[Move]:
        our_pieces = self.occupied_co[self.turn]
        # Generate moves for all our pieces.
        for from_square in scan_forward(our_pieces & from_mask):
            # Generate moves for all target squares.
            moves = self.moves(from_square) & to_mask
            for move in moves:
                yield Move(from_square, move)

    def is_legal(self, move: Move) -> bool:
        return move in self.generate_legal_moves()

    def is_capture(self, move: Move) -> bool:
        return move.to_square in self.captures(move.from_square)

    def is_game_over(self) -> bool:
        return self.outcome() is not None

    @property
    def winner(self) -> Optional[Color]:
        outcome = self.outcome()
        return outcome.winner if outcome else None

    @property
    def result(self) -> str:
        outcome = self.outcome()
        return outcome.result() if outcome else "*"

    def outcome(self) -> Optional[Outcome]:
        """Check if the game is over due to
        :func:`king_capture() <chess.Board.king_capture()>`,
        :func:`king_escape() <chess.Board.king_escape()>`, or
        :func:`is_stalemate() <chess.Board.is_stalemate()>`

        Alternatively use :func:`is_game_over() <chess.Board.is_game_over()>`
        to just check if the game is over.
        """
        if self.king_captured():
            return Outcome(termination=Termination.KING_CAPTURED, winner=BLACK)
        elif self.king_escaped():
            return Outcome(termination=Termination.KING_ESCAPED, winner=WHITE)
        elif self.is_stalemate():
            return Outcome(termination=Termination.STALEMATE, winner=None)
        else:
            return None

    def is_stalemate(self) -> bool:
        """Check if the game is over due to stalemate."""
        return self.fullmove_number >= self.move_limit or not self.legal_moves

    def king_captured(self) -> bool:
        """Check if the king is surrounded by enemy pieces."""
        king_bb = self.kings
        king_location = lsb(king_bb)

        surrounding_enemies = self.get_adjacent_opponent_pieces(king_location, WHITE)
        # if the king is surrounded by enemy pieces, return true
        if len(surrounding_enemies) == 4:
            return True
        if len(surrounding_enemies) != 3:
            return False
        # if the king is surrounded by 3 enemy pieces, check if the 4th empty square is the BB_THRONE
        surrounding_square = self.get_adjacent_empty_squares(king_location)
        return surrounding_square == BB_THRONE

    def king_escaped(self) -> bool:
        """Check if the king is at one of the four corners."""
        king_location = self.kings
        return bool(king_location & BB_CORNERS)

    def _board_state(self: BoardT) -> _BoardState[BoardT]:
        return _BoardState(self)

    def push(self: BoardT, move: Move) -> None:
        """
        Pushes a move to the move stack. The move is assumed to be legal.
        """
        board_state = self._board_state()
        self.move_stack.append(move)
        self._last_move[self.turn] = move
        self._stack.append(board_state)

        # increment the counters
        self.halfmove_clock += 1
        if self.turn == WHITE:
            self.fullmove_number += 1

        # on a null move, just flip the turn
        if not move:
            self.turn = not self.turn
            return

        # get the from and to squares
        piece_type = self.piece_at(move.from_square)
        assert (
            piece_type is not None
        ), f"no piece at {move.from_square}, {move}, {self.board_code()} \n {self}"

        captured_piece_types = None
        # check whether the move is a capture
        _, _, capture_map = self._captures_possible(move.from_square)
        # remove the piece at the from square
        self._remove_piece_at(move.from_square)
        # if the move is a capture, remove the piece at the to square
        # put the piece at the to square
        self._set_piece_at(move.to_square, piece_type.piece_type, self.turn)

        if move.to_square in capture_map.keys() and capture_map[move.to_square]:
            self.halfmove_clock = 0
            captured_piece_types = [
                self._remove_piece_at(sq)
                for sq in SquareSet(capture_map[move.to_square])
            ]

        if captured_piece_types:
            self._captured_pieces[self.turn].extend(captured_piece_types)

        # set the turn to the other player
        self.turn = not self.turn

    def push_code(self: BoardT, move_code: str) -> None:
        """Pushes a move to the move stack. The move is assumed to be legal."""
        self.push(Move.from_code(move_code))

    def pop(self: BoardT) -> Move:
        """Restores the previous position and returns the last move from the stack.

        Raises :exc:`IndexError` if the stack is empty.
        """
        move = self.move_stack.pop()
        self._stack.pop().restore(self)
        return move

    def peek(self) -> Move:
        """Returns the last move from the stack without restoring the position.

        Raises :exc:`IndexError` if the stack is empty.
        """
        return self.move_stack[-1]

    def find_move(self, from_square: Square, to_square: Square) -> Optional[Move]:
        """Finds a matching legal move from the given squares."""
        move = Move(from_square, to_square)
        moves = self.moves(from_square)
        if move in moves:
            return move
        else:
            raise IllegalMoveError(
                f"no matching legal move for {move.code} ({SQUARE_NAMES[from_square]} -> {SQUARE_NAMES[to_square]}) in {self.board_code()}"
            )

    def copy(self: BoardT, *, stack: Union[bool, int] = True) -> BoardT:
        """Returns a deep copy of the board."""
        board = super().copy()
        board.turn = self.turn
        board.fullmove_number = self.fullmove_number
        board.halfmove_clock = self.halfmove_clock
        if stack:
            stack = len(self.move_stack) if stack is True else stack
            board.move_stack = [copy.copy(move) for move in self.move_stack[-stack:]]
            board._stack = self._stack[-stack:]

        return board


class LegalMoveGenerator:
    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        return any(self.board.generate_legal_moves())

    def count(self) -> int:
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_legal_moves()

    def __contains__(self, move: Move) -> bool:
        return self.board.is_legal(move)

    def __repr__(self) -> str:
        codes = ", ".join(move.code for move in self)
        return f"<LegalMoveGenerator at {id(self):#x} ({codes})>"


IntoSquareSet = Union[SupportsInt, Iterable[Square]]


class SquareSet:
    """
    A set of squares.

    >>> import chess
    >>>
    >>> squares = chess.SquareSet([chess.A8, chess.A1])
    >>> squares
    SquareSet(0x0100_0000_0000_0001)

    >>> squares = chess.SquareSet(chess.BB_A8 | chess.BB_RANK_1)
    >>> squares
    SquareSet(0x0100_0000_0000_00ff)

    >>> print(squares)
    1 . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    1 1 1 1 1 1 1 1

    >>> len(squares)
    9

    >>> bool(squares)
    True

    >>> chess.B1 in squares
    True

    >>> for square in squares:
    ...     # 0 -- chess.A1
    ...     # 1 -- chess.B1
    ...     # 2 -- chess.C1
    ...     # 3 -- chess.D1
    ...     # 4 -- chess.E1
    ...     # 5 -- chess.F1
    ...     # 6 -- chess.G1
    ...     # 7 -- chess.H1
    ...     # 56 -- chess.A8
    ...     print(square)
    ...
    0
    1
    2
    3
    4
    5
    6
    7
    56

    >>> list(squares)
    [0, 1, 2, 3, 4, 5, 6, 7, 56]

    Square sets are internally represented by 121-bit integer masks of the
    included squares. Bitwise operations can be used to compute unions,
    intersections and shifts.

    >>> int(squares)
    72057594037928191

    >>> squares = SquareSet.from_mask(0x0100_0000_0000_0001)

    Also supports common set operations like
    :func:`~chess.SquareSet.issubset()`, :func:`~chess.SquareSet.issuperset()`,
    :func:`~chess.SquareSet.union()`, :func:`~chess.SquareSet.intersection()`,
    :func:`~chess.SquareSet.difference()`,
    :func:`~chess.SquareSet.symmetric_difference()` and
    :func:`~chess.SquareSet.copy()` as well as
    :func:`~chess.SquareSet.update()`,
    :func:`~chess.SquareSet.intersection_update()`,
    :func:`~chess.SquareSet.difference_update()`,
    :func:`~chess.SquareSet.symmetric_difference_update()` and
    :func:`~chess.SquareSet.clear()`.
    """

    def __init__(self, squares: IntoSquareSet = BB_EMPTY) -> None:
        try:
            self.mask = squares.__int__() & BB_ALL  # type: ignore
            return
        except AttributeError:
            self.mask = 0

        # Try squares as an iterable. Not under except clause for nicer
        # backtraces.
        for square in squares:  # type: ignore
            self.add(square)

    @classmethod
    def from_bb(cls, mask: Bitboard) -> "SquareSet":
        """Creates a square set from a bitboard mask."""
        s = cls()
        s.mask = mask
        return s

    # Set

    def __contains__(self, square: Square) -> bool:
        return bool(BB_SQUARES[square] & self.mask)

    def __iter__(self) -> Iterator[Square]:
        return scan_forward(self.mask)

    def __reversed__(self) -> Iterator[Square]:
        return scan_reversed(self.mask)

    def __len__(self) -> int:
        return popcount(self.mask)

    # MutableSet

    def add(self, square: Square) -> None:
        """Adds a square to the set."""
        self.mask |= BB_SQUARES[square]

    def discard(self, square: Square) -> None:
        """Discards a square from the set."""
        self.mask &= ~BB_SQUARES[square]

    # frozenset

    def isdisjoint(self, other: IntoSquareSet) -> bool:
        """Tests if the square sets are disjoint."""
        return not bool(self & other)

    def issubset(self, other: IntoSquareSet) -> bool:
        """Tests if this square set is a subset of another."""
        return not bool(self & ~SquareSet(other))

    def issuperset(self, other: IntoSquareSet) -> bool:
        """Tests if this square set is a superset of another."""
        return not bool(~self & other)

    def union(self, other: IntoSquareSet) -> "SquareSet":
        return self | other

    def __or__(self, other: IntoSquareSet) -> "SquareSet":
        r = SquareSet(other)
        r.mask |= self.mask
        return r

    def intersection(self, other: IntoSquareSet) -> "SquareSet":
        return self & other

    def __and__(self, other: IntoSquareSet) -> "SquareSet":
        r = SquareSet(other)
        r.mask &= self.mask
        return r

    def difference(self, other: IntoSquareSet) -> "SquareSet":
        return self - other

    def __sub__(self, other: IntoSquareSet) -> "SquareSet":
        r = SquareSet(other)
        r.mask = self.mask & ~r.mask
        return r

    def symmetric_difference(self, other: IntoSquareSet) -> "SquareSet":
        return self ^ other

    def __xor__(self, other: IntoSquareSet) -> "SquareSet":
        r = SquareSet(other)
        r.mask ^= self.mask
        return r

    def copy(self) -> "SquareSet":
        return SquareSet(self.mask)

    # set

    def update(self, *others: IntoSquareSet) -> None:
        for other in others:
            self |= other

    def __ior__(self, other: IntoSquareSet) -> "SquareSet":
        self.mask |= SquareSet(other).mask
        return self

    def intersection_update(self, *others: IntoSquareSet) -> None:
        for other in others:
            self &= other

    def __iand__(self, other: IntoSquareSet) -> "SquareSet":
        self.mask &= SquareSet(other).mask
        return self

    def difference_update(self, other: IntoSquareSet) -> None:
        self -= other

    def __isub__(self, other: IntoSquareSet) -> "SquareSet":
        self.mask &= ~SquareSet(other).mask
        return self

    def symmetric_difference_update(self, other: IntoSquareSet) -> None:
        self ^= other

    def __ixor__(self, other: IntoSquareSet) -> "SquareSet":
        self.mask ^= SquareSet(other).mask
        return self

    def remove(self, square: Square) -> None:
        """
        Removes a square from the set.

        :raises: :exc:`KeyError` if the given *square* was not in the set.
        """
        mask = BB_SQUARES[square]
        if self.mask & mask:
            self.mask ^= mask
        else:
            raise KeyError(square)

    def pop(self) -> Square:
        """
        Removes and returns a square from the set.

        :raises: :exc:`KeyError` if the set is empty.
        """
        if not self.mask:
            raise KeyError("pop from empty SquareSet")

        square = lsb(self.mask)
        self.mask &= self.mask - 1
        return square

    def clear(self) -> None:
        """Removes all elements from this set."""
        self.mask = BB_EMPTY

    # SquareSet

    def carry_rippler(self) -> Iterator[Bitboard]:
        """Iterator over the subsets of this set."""
        return _carry_rippler(self.mask)

    def mirror(self, vertical=True) -> "SquareSet":
        """Returns a vertically mirrored copy of this square set."""
        return (
            SquareSet(flip_vertical(self.mask))
            if vertical
            else SquareSet(flip_horizontal(self.mask))
        )

    def tolist(self) -> List[bool]:
        """Converts the set to a list of 121 bools."""
        result = [False] * 121
        for square in self:
            result[square] = True
        return result

    def __bool__(self) -> bool:
        return bool(self.mask)

    def __eq__(self, other: object) -> bool:
        try:
            return self.mask == SquareSet(other).mask  # type: ignore
        except (TypeError, ValueError):
            return NotImplemented

    def __lshift__(self, shift: int) -> "SquareSet":
        return SquareSet((self.mask << shift) & BB_ALL)

    def __rshift__(self, shift: int) -> "SquareSet":
        return SquareSet(self.mask >> shift)

    def __ilshift__(self, shift: int) -> "SquareSet":
        self.mask = (self.mask << shift) & BB_ALL
        return self

    def __irshift__(self, shift: int) -> "SquareSet":
        self.mask >>= shift
        return self

    def __invert__(self) -> "SquareSet":
        return SquareSet(~self.mask & BB_ALL)

    def __int__(self) -> int:
        return self.mask

    def __index__(self) -> int:
        return self.mask

    def __repr__(self) -> str:
        return f"SquareSet({self.mask:#021_x})"

    def __str__(self) -> str:
        builder = []
        for square in SQUARES:
            mask = BB_SQUARES[square]
            builder.append("1" if self.mask & mask else ".")

            if not mask & BB_FILE_K:
                builder.append(" ")
            elif square != K11:
                builder.append("\n")

        return "".join(builder)

    @classmethod
    def ray(cls, a: Square, b: Square) -> "SquareSet":
        """
        All squares on the rank, file or diagonal with the two squares, if they
        are aligned.

        >>> import chess
        >>>
        >>> print(chess.SquareSet.ray(chess.E2, chess.B5))
        . . . . . . . .
        . . . . . . . .
        1 . . . . . . .
        . 1 . . . . . .
        . . 1 . . . . .
        . . . 1 . . . .
        . . . . 1 . . .
        . . . . . 1 . .
        """
        return cls(ray(a, b))

    @classmethod
    def between(cls, a: Square, b: Square) -> "SquareSet":
        """
        All squares on the rank, file or diagonal between the two squares
        (bounds not included), if they are aligned.

        >>> import chess
        >>>
        >>> print(chess.SquareSet.between(chess.E2, chess.B5))
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . 1 . . . . .
        . . . 1 . . . .
        . . . . . . . .
        . . . . . . . .
        """
        return cls(between(a, b))

    @classmethod
    def from_square(cls, square: Square) -> "SquareSet":
        """
        Creates a :class:`~chess.SquareSet` from a single square.

        >>> import chess
        >>>
        >>> chess.SquareSet.from_square(chess.A1) == chess.BB_A1
        True
        """
        return cls(BB_SQUARES[square])


class KingCapturedEasierBoard(Board):

    # Board class variant that makes it easier to check if the king is captured.

    def __init__(self: BoardT, code: Optional[str] = STARTING_POSITION_CODE) -> None:
        super().__init__(code)
        self.fullmove_number = 1
        self.halfmove_clock = 0
        self.strict = True

    def generate_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        """If there is a move that will end the game, only return those moves if strict is True."""
        if not self.strict:
            yield from super().generate_legal_moves(from_mask, to_mask)
            return

        for move in super().generate_legal_moves(from_mask, to_mask):
            self.push(move)
            if self.king_captured() or self.king_escaped():
                yield self.pop()
                break
            else:
                self.pop()
        else:
            yield from super().generate_legal_moves(from_mask, to_mask)

    def king_captured(self) -> bool:
        """Check if the king is captured either by a sandwich outside of his throne,
        or by being surrounded by 4 enemies on the throne.
        """
        bb_king = self.kings
        bb_up_from_king = shift_up(bb_king)
        bb_down_from_king = shift_down(bb_king)
        bb_left_from_king = shift_left(bb_king)
        bb_right_from_king = shift_right(bb_king)

        # was the previous black move to a space next to the king?
        if BB_SQUARES[self._last_move[BLACK].to_square] & ~(
            bb_up_from_king | bb_down_from_king | bb_left_from_king | bb_right_from_king
        ):
            return False

        # Is the king on the throne?
        if bb_king & BB_THRONE:
            return self._check_king_surrounded_on_throne(
                bb_king,
                bb_up_from_king,
                bb_down_from_king,
                bb_left_from_king,
                bb_right_from_king,
            )
        else:
            return self._check_king_sandwich(
                bb_king,
                bb_up_from_king,
                bb_down_from_king,
                bb_left_from_king,
                bb_right_from_king,
            )

    def _check_king_surrounded_on_throne(
        self,
        bb_king,
        bb_up_from_king,
        bb_down_from_king,
        bb_left_from_king,
        bb_right_from_king,
    ):
        # Is the king surrounded by 4 enemies on the throne?
        return (
            bool(bb_up_from_king & self.occupied_co[BLACK])
            and bool(bb_down_from_king & self.occupied_co[BLACK])
            and bool(bb_left_from_king & self.occupied_co[BLACK])
            and bool(bb_right_from_king & self.occupied_co[BLACK])
        )

    def _check_king_sandwich(
        self,
        bb_king,
        bb_up_from_king,
        bb_down_from_king,
        bb_left_from_king,
        bb_right_from_king,
    ):
        # Is the king sandwiched between 2 enemies outside of his throne?
        return bool(
            bb_left_from_king & self.occupied_co[BLACK]
            and bb_right_from_king & self.occupied_co[BLACK]
            or bb_up_from_king & self.occupied_co[BLACK]
            and bb_down_from_king & self.occupied_co[BLACK]
        )


class KingEscapeAndCaptureEasierBoard(KingCapturedEasierBoard):
    def __init__(
        self: BoardT,
        code: Optional[str] = STARTING_POSITION_CODE,
        strict: bool = False,
    ) -> None:
        super().__init__(code)
        self.fullmove_number = 1
        self.halfmove_clock = 0
        self.strict = strict

    def king_escaped(self) -> bool:
        return bool(self.kings & BB_EDGE)

    def generate_legal_moves(
        self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL
    ) -> Iterator[Move]:
        """If there is a move that will end the game, only return those moves if strict is True."""
        if not self.strict:
            yield from super().generate_legal_moves(from_mask, to_mask)
            return

        for move in super().generate_legal_moves(from_mask, to_mask):
            self.push(move)
            if self.king_captured() or self.king_escaped():
                yield self.pop()
                break
            else:
                self.pop()
        else:
            yield from super().generate_legal_moves(from_mask, to_mask)

import random

def generate_move(mode='random'):
    if mode == 'random':
        return generate_random_move
    elif mode == 'always_cap':
        return generate_move_always_capture_if_possible
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
def generate_random_move(board: BoardT) -> Move:
    return random.choice(list(board.legal_moves))

def generate_move_always_capture_if_possible(board: BoardT) -> Move:
    caps = board._all_capture_moves(board.turn)
    if caps:
        return random.choice(list(caps))
    else:
        return generate_random_move(board)


def generate_random_game(_, black_move_mode='random', white_move_mode='random'):
    Board.move_limit = 500
    board = KingEscapeAndCaptureEasierBoard(strict=True)
    # play random moves
    while not board.is_game_over():
        if board.turn == BLACK:
            move = generate_move(black_move_mode)(board)
        else:
            move = generate_move(white_move_mode)(board)
        board.push(move)
    return board

# set move moves without losing the ability to pickle the generate_random_game function for multiprocessing
generate_random_game_black_always_captures = partial(generate_random_game, black_move_mode='always_cap', white_move_mode='random')
generate_random_game_white_always_captures = partial(generate_random_game, black_move_mode='random', white_move_mode='always_cap')
generate_random_game_always_captures = partial(generate_random_game, black_move_mode='always_cap', white_move_mode='always_cap')

def generate_games(num=10000, game_func=generate_random_game):
    from tqdm import tqdm
    # multiprocessing
    from multiprocessing import Pool

    # generate a bunch of games
    games = []
    # multiprocessing with tqdm
    with Pool() as p:
        games.extend(
            iter(tqdm(p.imap_unordered(game_func, range(num)), total=num))
        )
    return games


if __name__ == "__main__":

    """
    A B C D E F G H I J K
    . . . 1 1 1 1 1 . . . 1
    . . . . . 1 . . . . . 2
    . . . . . . . . . . . 3
    1 . . . . 1 . . . . 1 4
    1 . . . 1 1 1 . . . 1 5
    1 1 . 1 1 K 1 1 . 1 1 6
    1 . . . 1 1 1 . . . 1 7
    1 . . . . 1 . . . . 1 8
    . . . . . . . . . . . 9
    . . . . . 1 . . . . . +
    . . . 1 1 1 1 1 . . . *
    """
    from hnefatafl import *
    games = generate_games(100, generate_random_game_always_captures)
    board = KingEscapeAndCaptureEasierBoard(strict=False)
    for game in games:
        if game.outcome().winner is not None:
            for state in game._stack:
                state.restore(board)
                print(board)
                if board.is_game_over():
                    print(board.outcome())
                    sleep(5)
                    break
                sleep(0.5)
                # clear the output
                import os
                os.system("cls" if os.name == "nt" else "clear")
            break

    # summarize the games
    winners = [game.outcome().winner for game in games]
    print("Number of games:", len(games))
    print("Number of white wins:", winners.count(WHITE))
    print("Number of black wins:", winners.count(BLACK))
    print("Number of draws:", winners.count(None))
    # average number of moves
    print("Average number of moves:", sum(len(game.move_stack) for game in games) / len(games))
    # average number of captures
    print("Average number of captures:", sum(len(game._captured_pieces[WHITE]) + len(game._captured_pieces[BLACK]) for game in games) / len(games))
    # average number of captures per color
    print("Average number of white captures:", sum(len(game._captured_pieces[WHITE]) for game in games) / len(games))
    print("Average number of black captures:", sum(len(game._captured_pieces[BLACK]) for game in games) / len(games))

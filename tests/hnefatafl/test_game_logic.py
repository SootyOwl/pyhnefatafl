"""Test the rules of the game.

Rules to test:
# Movement
- Pieces can move horizontally or vertically, but not diagonally.
- Pieces cannot move through or jump other pieces.
- Men cannot move into a corner.
- Men cannot move onto the central square.
- The king can move onto the central square.
- The king can move into a corner, which ends the game.

# Capture
- A piece is captured when it is sandwiched between two opposing pieces, provided one of the opposing pieces moved to capture it.
- A piece moving into a sandwiched position is not captured.
- A piece is not captured if it is sandwiched between two friendly pieces.
- A piece is not captured if it is sandwiched between two opposing pieces, but neither of the opposing pieces moved to capture it.
- The king on the central square is only captured if is surrounded by four opposing pieces.
- The king adjacent to the central square is only captured if is surrounded by three opposing pieces and the central square.
- The king outside the central square is captured like any other piece, by being sandwiched.

# Win
- The king escapes when it moves to the edge of the board, and white wins.
- The king escapes when it moves to a corner, and white wins.
- The king is captured when it is sandwiched between four opposing pieces on the throne, and black wins.
- The king is captured when it is sandwiched between three opposing pieces and the throne, and black wins.
- The king is captured when it is sandwiched between two opposing pieces and is not next to or on the throne, and black wins.

# Draw
- The game is a draw if a player cannot move any of their pieces on their turn.
- The game is a draw if the move limit is reached.
"""

from typing import Union
import pytest
from pytest import param
from hnefatafl import (
    FILE_NAMES,
    RANK_NAMES,
    KingEscapeAndCaptureEasierBoard,
    Move,
    Piece,
    Termination,
    square,
    parse_square,
    MAN,
    KING,
    WHITE,
    BLACK,
)


@pytest.fixture
def board():
    return KingEscapeAndCaptureEasierBoard(strict=False)


@pytest.fixture
def empty_board(board: KingEscapeAndCaptureEasierBoard):
    board.clear_board()
    return board


# Movement
@pytest.mark.parametrize("piece_type", [MAN, KING])
@pytest.mark.parametrize("color", [WHITE, BLACK])
@pytest.mark.parametrize(
    "offset,legal",
    [
        param((0, 1), True, id="up"),
        param((0, -1), True, id="down"),
        param((1, 0), True, id="right"),
        param((-1, 0), True, id="left"),
        param((1, 1), False, id="diagonal1"),
        param((1, -1), False, id="diagonal2"),
        param((-1, 1), False, id="diagonal3"),
        param((-1, -1), False, id="diagonal4"),
    ],
)
def test_piece_moves(
    empty_board: KingEscapeAndCaptureEasierBoard,
    piece_type: str,
    color: bool,
    offset: tuple[int, int],
    legal: bool,
):
    """Test that a piece can move along a file."""
    empty_board.set_piece_at(square(4, 4), Piece(color=color, piece_type=piece_type))
    empty_board.turn = color
    move = Move(square(4, 4), square(4 + offset[0], 4 + offset[1]))
    assert (
        empty_board.is_legal(move) == legal
    ), f'Move {move} should{" " if legal else " not "}be legal.'


# test piece cannot move through (or jump) other pieces
@pytest.mark.parametrize("piece_type", [MAN, KING])
@pytest.mark.parametrize("color", [WHITE, BLACK])
@pytest.mark.parametrize("blocking_piece_type", [MAN, KING])
@pytest.mark.parametrize("blocking_color", [WHITE, BLACK])
def test_piece_cannot_move_through_pieces(
    empty_board: KingEscapeAndCaptureEasierBoard,
    piece_type: str,
    color: bool,
    blocking_piece_type: str,
    blocking_color: str,
):
    """Test that a piece cannot move through (or jump) other pieces."""
    empty_board.set_piece_at(square(4, 4), Piece(color=color, piece_type=piece_type))
    empty_board.set_piece_at(
        square(4, 5), Piece(color=blocking_color, piece_type=blocking_piece_type)
    )
    empty_board.turn = color
    move = Move(square(4, 4), square(4, 6))
    assert empty_board.is_legal(move) is False, f"Move {move} should not be legal."


@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_men_cannot_move_into_corner(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    """"""
    empty_board.set_piece_at(square(0, 6), Piece(color=color, piece_type=MAN))
    empty_board.turn = color
    move = Move(square(0, 6), square(0, 0))
    assert empty_board.is_legal(move) is False, f"Move {move} should not be legal."


@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_men_cannot_move_into_center(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    """Test men cannot move into the center square (F6)."""
    empty_board.set_piece_at(
        parse_square("h6"), Piece.from_symbol("M" if color else "m")
    )
    empty_board.turn = color
    move = Move(parse_square("h6"), parse_square("f6"))
    assert empty_board.is_legal(move) is False, f"Move {move} should not be legal."

    # piece can move over the center square
    move = Move(parse_square("h6"), parse_square("a6"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."


def test_king_can_move_into_center(empty_board: KingEscapeAndCaptureEasierBoard):
    """Test the king can move into the center square (F6)."""
    empty_board.set_piece_at(parse_square("h6"), Piece.from_symbol("K"))
    empty_board.turn = WHITE
    move = Move(parse_square("h6"), parse_square("f6"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."


def test_king_can_move_into_corner(empty_board: KingEscapeAndCaptureEasierBoard):
    """Test that the king can move onto the corner squares."""
    empty_board.set_piece_at(parse_square("k6"), Piece.from_symbol("K"))
    empty_board.turn = WHITE
    move = Move(parse_square("k6"), parse_square("k#"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."


# Capture
@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_piece_is_captured_when_sandwiched_between_enemy(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("M" if color else "m"),
            parse_square("e5"): Piece.from_symbol("m" if color else "M"),
            parse_square("f6"): Piece.from_symbol("M" if color else "m"),
        }
    )
    empty_board.turn = color
    move = Move(parse_square("f6"), parse_square("f5"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("f6")) is None
    ), f'Piece at {parse_square("f6")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e5")) is None
    ), f'Piece at {parse_square("e5")} should have been captured.'


@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_piece_is_not_captured_when_not_sandwiched_between_enemy(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("M" if color else "m"),
            parse_square("e5"): Piece.from_symbol("m" if color else "M"),
            parse_square("f6"): Piece.from_symbol("M" if color else "m"),
        }
    )
    empty_board.turn = color
    move = Move(parse_square("f6"), parse_square("f4"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("f6")) is None
    ), f'Piece at {parse_square("f6")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e5")) is not None
    ), f'Piece at {parse_square("e5")} should not have been captured.'


@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_piece_is_not_captured_when_sandwiched_between_own(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("M" if color else "m"),
            parse_square("e5"): Piece.from_symbol("M" if color else "m"),
            parse_square("f6"): Piece.from_symbol("M" if color else "m"),
        }
    )
    empty_board.turn = color
    move = Move(parse_square("f6"), parse_square("f5"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("f6")) is None
    ), f'Piece at {parse_square("f6")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e5")) is not None
    ), f'Piece at {parse_square("e5")} should not have been captured.'


@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_king_can_capture(empty_board: KingEscapeAndCaptureEasierBoard, color: bool):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("M" if color else "m"),
            parse_square("e5"): Piece.from_symbol("m" if color else "M"),
            parse_square("f6"): Piece.from_symbol("K" if color else "m"),
        }
    )
    empty_board.turn = color
    move = Move(parse_square("f6"), parse_square("f5"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("f6")) is None
    ), f'Piece at {parse_square("f6")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e5")) is None
    ), f'Piece at {parse_square("e5")} should have been captured.'


# test moving into sandwiched position
@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_piece_can_move_into_sandwiched_position_without_being_captured(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("m" if color else "M"),
            parse_square("e7"): Piece.from_symbol("M" if color else "m"),
            parse_square("f5"): Piece.from_symbol("m" if color else "M"),
        }
    )
    empty_board.turn = color
    move = Move(parse_square("e7"), parse_square("e5"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("e7")) is None
    ), f'Piece at {parse_square("e7")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e5")) is not None
    ), f'Piece at {parse_square("e5")} should not have been captured.'


# test moving into sandwiched position as the king
def test_king_can_move_into_sandwiched_position_without_being_captured(
    empty_board: KingEscapeAndCaptureEasierBoard,
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("m"),
            parse_square("e7"): Piece.from_symbol("K"),
            parse_square("f5"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = WHITE
    move = Move(parse_square("e7"), parse_square("e5"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("e7")) is None
    ), f'Piece at {parse_square("e7")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e5")) is not None
    ), f'Piece at {parse_square("e5")} should not have been captured.'


# Test that pieces can be captured against the throne as long as the king is not there
@pytest.mark.parametrize("color", [WHITE, BLACK])
def test_piece_can_be_captured_against_throne(
    empty_board: KingEscapeAndCaptureEasierBoard, color: bool
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d7"): Piece.from_symbol("M" if color else "m"),
            parse_square("e6"): Piece.from_symbol("m" if color else "M"),
        }
    )
    empty_board.turn = color
    move = Move(parse_square("d7"), parse_square("d6"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("d7")) is None
    ), f'Piece at {parse_square("d7")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e6")) is None
    ), f'Piece at {parse_square("e6")} should have been captured.'


# Test that white pieces cannot be captured against the throne if the king is there
def test_white_piece_cannot_be_captured_against_throne_if_king_is_there(
    empty_board: KingEscapeAndCaptureEasierBoard,
):
    # set up board with piece to be sandwiched
    empty_board.set_piece_map(
        {
            parse_square("d7"): Piece.from_symbol("m"),
            parse_square("e6"): Piece.from_symbol("M"),
            parse_square("f6"): Piece.from_symbol("K"),
        }
    )
    empty_board.turn = BLACK
    move = Move(parse_square("d7"), parse_square("d6"))

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("d7")) is None
    ), f'Piece at {parse_square("d7")} should have moved.'
    assert (
        empty_board.piece_at(parse_square("e6")) is not None
    ), f'Piece at {parse_square("e6")} should not have been captured.'


# Win
@pytest.mark.parametrize("edge", ["A", "K", "1", "#"])
def test_king_escape_endgame(
    empty_board: KingEscapeAndCaptureEasierBoard, edge: Union[str, int]
):
    """Test that the game ends with a White win when the King reaches the edge of the board."""
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("m"),
            parse_square("e7"): Piece.from_symbol("K"),
            parse_square("f5"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = WHITE
    if edge in FILE_NAMES:
        move = Move(parse_square("e7"), parse_square(f"{edge}7"))
    elif edge in RANK_NAMES:
        move = Move(parse_square("e7"), parse_square(f"e{edge}"))

    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("e7")) is None
    ), f'King at {parse_square("e7")} should have moved.'

    outcome = empty_board.outcome()
    assert outcome is not None
    assert outcome.winner == WHITE
    assert outcome.termination == Termination.KING_ESCAPED


# test that the game ends with a Black win when the King is captured by two black pieces outside of the King's throne
def test_king_captured_outside_throne(empty_board: KingEscapeAndCaptureEasierBoard):
    empty_board.set_piece_map(
        # add a king to the board outside of the throne and two black pieces to capture it
        {
            parse_square("b5"): Piece.from_symbol("m"),
            parse_square("b6"): Piece.from_symbol("K"),
            parse_square("c7"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = BLACK
    move = Move(parse_square("c7"), parse_square("b7"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("c7")) is None
    ), f'Piece at {parse_square("c7")} should have moved.'

    outcome = empty_board.outcome()
    assert (
        outcome is not None
    ), f"Outcome should not be None: King should have been captured. {outcome}, {empty_board}"
    assert outcome.winner == BLACK
    assert outcome.termination == Termination.KING_CAPTURED


# test that the game ends with a Black win when the King on the throne and surrounded by four black pieces
def test_king_captured_on_throne(empty_board: KingEscapeAndCaptureEasierBoard):
    empty_board.set_piece_map(
        # add a king to the throne and four black pieces to capture it
        {
            parse_square("f6"): Piece.from_symbol("K"),
            parse_square("e6"): Piece.from_symbol("m"),
            parse_square("g6"): Piece.from_symbol("m"),
            parse_square("f5"): Piece.from_symbol("m"),
            parse_square("e7"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = BLACK
    move = Move(parse_square("e7"), parse_square("f7"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("e7")) is None
    ), f'Piece at {parse_square("e7")} should have moved.'

    outcome = empty_board.outcome()
    assert (
        outcome is not None
    ), f"Outcome should not be None: King should have been captured. {outcome}, {empty_board}"
    assert outcome.winner == BLACK
    assert outcome.termination == Termination.KING_CAPTURED


# test that the game ends with a Black win when the King is adjacent to the throne (f6) and surrounded by three black pieces on all other sides
def test_king_captured_adjacent_to_throne(empty_board: KingEscapeAndCaptureEasierBoard):
    empty_board.set_piece_map(
        # add a king next to the throne and three black pieces to capture it
        {
            parse_square("e6"): Piece.from_symbol("K"),
            parse_square("e5"): Piece.from_symbol("m"),
            parse_square("e7"): Piece.from_symbol("m"),
            parse_square("d5"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = BLACK
    move = Move(parse_square("d5"), parse_square("d6"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("d5")) is None
    ), f'Piece at {parse_square("d5")} should have moved.'

    outcome = empty_board.outcome()
    assert (
        outcome is not None
    ), f"Outcome should not be None: King should have been captured. {outcome}, {empty_board}"
    assert outcome.winner == BLACK
    assert outcome.termination == Termination.KING_CAPTURED


# Draw
# test that the game ends in a draw if the move limit is reached
def test_move_limit_reached(empty_board: KingEscapeAndCaptureEasierBoard):
    empty_board.move_limit = 1
    empty_board.set_piece_map(
        {
            parse_square("d5"): Piece.from_symbol("m"),
            parse_square("e7"): Piece.from_symbol("K"),
            parse_square("f5"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = WHITE
    move = Move(parse_square("e7"), parse_square("e8"))
    assert empty_board.is_legal(move) is True, f"Move {move} should be legal."

    # make the move
    empty_board.push(move)
    assert (
        empty_board.piece_at(parse_square("e7")) is None
    ), f'King at {parse_square("e7")} should have moved.'

    outcome = empty_board.outcome()
    assert outcome is not None
    assert outcome.winner is None
    assert outcome.termination == Termination.STALEMATE


# test that the game ends in a draw if there are no legal moves for the current side (i.e Black loses all pieces)
def test_no_legal_moves(empty_board: KingEscapeAndCaptureEasierBoard):
    empty_board.set_piece_map(
        {
            parse_square("e7"): Piece.from_symbol("K"),
        }
    )
    empty_board.turn = BLACK
    outcome = empty_board.outcome()
    assert outcome is not None
    assert outcome.winner is None
    assert outcome.termination == Termination.STALEMATE


# Test strict mode, which forces the game to end if there is a move which would result in a king capture or escape
def test_strict_mode_king_capture(empty_board: KingEscapeAndCaptureEasierBoard):
    """Test that if the king can be captured, that is the only legal move in strict mode."""
    empty_board.set_piece_map(
        {
            parse_square("e7"): Piece.from_symbol("K"),
            parse_square("e6"): Piece.from_symbol("m"),
            parse_square("e8"): Piece.from_symbol("m"),
            parse_square("d6"): Piece.from_symbol("m"),
            parse_square("f7"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = BLACK
    empty_board.strict = True
    # get the legal moves
    legal_moves = list(empty_board.legal_moves)
    assert len(legal_moves) == 1
    assert legal_moves[0].from_square == parse_square("d6")
    assert legal_moves[0].to_square == parse_square("d7")

    empty_board.push(legal_moves[0])
    outcome = empty_board.outcome()
    assert outcome is not None
    assert outcome.winner == BLACK
    assert outcome.termination == Termination.KING_CAPTURED


def test_strict_mode_king_escape(empty_board: KingEscapeAndCaptureEasierBoard):
    """Test that if the king can escape, that is the only legal move in strict mode."""
    empty_board.set_piece_map(
        {
            parse_square("e7"): Piece.from_symbol("K"),
            parse_square("e6"): Piece.from_symbol("m"),
            parse_square("e8"): Piece.from_symbol("m"),
            parse_square("d6"): Piece.from_symbol("m"),
            parse_square("f7"): Piece.from_symbol("m"),
        }
    )
    empty_board.turn = WHITE
    empty_board.strict = True
    # get the legal moves
    legal_moves = list(empty_board.legal_moves)
    assert len(legal_moves) == 1
    assert legal_moves[0].from_square == parse_square("e7")
    assert legal_moves[0].to_square == parse_square("a7")

    empty_board.push(legal_moves[0])
    assert empty_board.outcome() is not None
    assert empty_board.outcome().winner == WHITE
    

# test that just plays random sample games (legal moves only) and ensures nothing goes wrong
def test_play_random_games(board: KingEscapeAndCaptureEasierBoard):
    import random

    random.seed(0)
    played = 0
    for _ in range(100):
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)
        played += 1
        board.reset()
    assert played == 100

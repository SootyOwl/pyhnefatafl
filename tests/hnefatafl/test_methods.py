# test board.copy()
import pytest
import hnefatafl as hn

@pytest.fixture
def board():
    return hn.KingEscapeAndCaptureEasierBoard()


def test_copy_board_is_deep(board: hn.BoardT):
    """Test that board.copy() makes a deep copy of the board."""
    board2 = board.copy()
    
    # assert equality
    assert board2 == board
    assert board.get_code() == board2.get_code(), "Board codes should be equal."
    assert board.occupied == board2.occupied, "Occupied squares should be equal."
    assert board.occupied_co == board2.occupied_co, "Occupied squares should be equal."
    assert board.occupied_co[hn.WHITE] == board2.occupied_co[hn.WHITE], "Occupied squares should be equal."
    assert board.occupied_co[hn.BLACK] == board2.occupied_co[hn.BLACK], "Occupied squares should be equal."
    assert board.kings == board2.kings, "King position should be equal."
    assert board.turn == board2.turn, "Turn should be equal."
    

    # assert not same object
    assert board is not board2, "board.copy() should return a new object."
    assert board.occupied is not board2.occupied, "board.copy() should return a new object."
    assert board.occupied_co is not board2.occupied_co, "board.copy() should return a new object."
    assert board.occupied_co[hn.WHITE] is not board2.occupied_co[hn.WHITE], "board.copy() should return a new object."
    assert board.occupied_co[hn.BLACK] is not board2.occupied_co[hn.BLACK], "board.copy() should return a new object."
    assert board.kings is not board2.kings, "board.copy() should return a new object."
    # turn is just a bool, so it should be the same object
    assert board.turn is board2.turn, "board.copy() should return a new object."

    # assert that changing board2 does not change board
    board2.push(hn.Move.from_code("A5.4"))  # black moves
    board2.push(hn.Move.from_code("E5.4"))  # white moves
    assert board2 != board, "Changing board2 should not change board."
    assert board.get_code() != board2.get_code(), "Changing board2 should not change board."
    assert board.occupied != board2.occupied, "Changing board2 should not change board."
    assert board.occupied_co != board2.occupied_co, "Changing board2 should not change board."
    assert board.occupied_co[hn.BLACK] != board2.occupied_co[hn.BLACK], "Changing board2 should not change board."
    assert board.occupied_co[hn.WHITE] != board2.occupied_co[hn.WHITE], "Changing board2 should not change board."


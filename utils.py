"""Board <-> tensor helpers and small utilities."""
import chess
import numpy as np

NUM_PLANES = 18


def board_to_planes(board):
    """Convert a python-chess board into an 18 x 8 x 8 float array.

    Planes:
        0..5    white P,N,B,R,Q,K
        6..11   black P,N,B,R,Q,K
        12..15  castling rights (W-K, W-Q, B-K, B-Q), all-ones when available
        16      en passant square
        17      side to move (all-ones for white)
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for color in (chess.WHITE, chess.BLACK):
        offset = 0 if color == chess.WHITE else 6
        for piece_type in chess.PIECE_TYPES:
            plane_idx = offset + (piece_type - 1)
            for sq in board.pieces(piece_type, color):
                planes[plane_idx, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12].fill(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13].fill(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14].fill(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15].fill(1.0)

    if board.ep_square is not None:
        planes[16, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0

    if board.turn == chess.WHITE:
        planes[17].fill(1.0)

    return planes


def result_to_z(result, perspective_color):
    """Game result string -> +1/-1/0 from the perspective of a color."""
    if result == "1/2-1/2":
        return 0.0
    white_wins = result == "1-0"
    if perspective_color == chess.WHITE:
        return 1.0 if white_wins else -1.0
    return -1.0 if white_wins else 1.0


def load_checkpoint(path, map_location="cpu"):
    """torch.load that tolerates PyTorch >=2.6 weights_only defaults.

    Our local checkpoints are trusted; retry with the permissive unpickler
    when the strict loader rejects extra object types (e.g. numpy arrays).
    """
    import torch
    try:
        return torch.load(path, map_location=map_location)
    except Exception:  # noqa: BLE001
        return torch.load(path, map_location=map_location, weights_only=False)

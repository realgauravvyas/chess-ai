"""Lichess-style 8x8x73 move encoding (4672 total actions).

For each of the 64 squares there are 73 action planes:

    0..55   queen moves: 8 directions x 7 step lengths
    56..63  knight moves (8 deltas)
    64..72  underpromotions: 3 pieces (knight, bishop, rook) x 3 directions
            (direction 0 = left, 1 = straight, 2 = right, relative to mover)

An action index is:  index = from_square * 73 + plane.
"""
import chess

NUM_POLICY = 64 * 73
NUM_ACTION_PLANES = 73

# (file_delta, rank_delta) for the 8 queen directions
QUEEN_DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
]
QUEEN_DIRECTION_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}

# 8 knight deltas
KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
KNIGHT_DELTA_INDEX = {d: i for i, d in enumerate(KNIGHT_DELTAS)}

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# left-right mirror remaps (chess is symmetric across the vertical axis)
_QUEEN_MIRROR = [4, 3, 2, 1, 0, 7, 6, 5]   # direction d -> mirrored direction
_KNIGHT_MIRROR = [7, 6, 5, 4, 3, 2, 1, 0]  # delta (-df, dr)
_PROMO_MIRROR = [2, 1, 0]                  # left <-> right, straight stays


def mirror_square(square):
    return chess.square(7 - chess.square_file(square),
                        chess.square_rank(square))


def mirror_action_index(index):
    """Mirror an action index across the board's vertical axis.

    Mirroring a position (file f -> 7-f) maps every legal action to the
    mirrored action of the mirrored position; this is its inverse-free map.
    """
    square, plane = divmod(index, NUM_ACTION_PLANES)
    new_sq = mirror_square(square)

    if plane < 56:      # queen moves: plane = direction*7 + (step-1)
        direction, step = divmod(plane, 7)
        new_plane = _QUEEN_MIRROR[direction] * 7 + step
    elif plane < 64:    # knight moves
        new_plane = 56 + _KNIGHT_MIRROR[plane - 56]
    else:               # underpromotions: piece*3 + dir(0=left,1=st,2=right)
        up = plane - 64
        piece, direction = up // 3, up % 3
        new_plane = 64 + piece * 3 + _PROMO_MIRROR[direction]
    return new_sq * NUM_ACTION_PLANES + new_plane


def encode_move(move):
    """Map a chess.Move to an integer index in [0, 4672)."""
    fs = move.from_square
    ff, fr = chess.square_file(fs), chess.square_rank(fs)
    tf, tr = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    df, dr = tf - ff, tr - fr
    base = fs * NUM_ACTION_PLANES

    # underpromotions
    if move.promotion and move.promotion != chess.QUEEN:
        piece = UNDERPROMO_PIECES.index(move.promotion)
        if df == 0:
            direction = 1  # straight
        else:
            moving_up = dr > 0
            left = (df < 0) if moving_up else (df > 0)
            direction = 0 if left else 2
        return base + 64 + piece * 3 + direction

    # knight moves (absolute deltas work for both colors)
    if (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        return base + 56 + KNIGHT_DELTA_INDEX[(df, dr)]

    # everything else (including queen promotions) is a queen-style move
    step = max(abs(df), abs(dr))
    direction = QUEEN_DIRECTION_INDEX[(df // step, dr // step)]
    return base + direction * 7 + (step - 1)


def decode_move(index):
    """Inverse of encode_move. Debugging helper only.

    Note: a queen promotion shares its plane with other queen moves onto the
    last rank, so the promotion flag is guessed when the move lands on a
    promotion square (rank 8 for white / rank 1 for black pawns).
    """
    square = index // NUM_ACTION_PLANES
    plane = index % NUM_ACTION_PLANES
    ff, fr = chess.square_file(square), chess.square_rank(square)

    if plane < 56:
        direction, step = divmod(plane, 7)
        df, dr = QUEEN_DIRECTIONS[direction]
        tf, tr = ff + (step + 1) * df, fr + (step + 1) * dr
        to_square = chess.square(tf, tr)
        if (fr == 6 and tr == 7) or (fr == 1 and tr == 0):
            return chess.Move(square, to_square, promotion=chess.QUEEN)
        return chess.Move(square, to_square)

    if plane < 64:
        df, dr = KNIGHT_DELTAS[plane - 56]
        return chess.Move(square, chess.square(ff + df, fr + dr))

    uprom = plane - 64
    piece = UNDERPROMO_PIECES[uprom // 3]
    direction = uprom % 3
    moving_up = fr == 6
    df = (-1, 0, 1)[direction] if moving_up else (1, 0, -1)[direction]
    dr = 1 if moving_up else -1
    return chess.Move(square, chess.square(ff + df, fr + dr), promotion=piece)


def _self_test():
    import random

    boards = [chess.Board()]
    for _ in range(20):
        b = chess.Board()
        for _ in range(random.randrange(5, 40)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))
        boards.append(b)

    errors = 0
    checked = 0
    for b in boards:
        seen = {}
        for move in b.legal_moves:
            checked += 1
            idx = encode_move(move)
            if idx in seen and seen[idx] != move:
                print(f"Collision: {seen[idx]} vs {move} -> {idx}")
                errors += 1
            seen[idx] = move
        for idx, move in seen.items():
            decoded = decode_move(idx)
            base = chess.Move(decoded.from_square, decoded.to_square)
            if base != chess.Move(move.from_square, move.to_square):
                print(f"Mismatch: {move} -> {idx} -> {decoded}")
                errors += 1

        # mirror consistency: mirrored action == encode(mirrored move)
        for move in b.legal_moves:
            idx = encode_move(move)
            mf = mirror_square(move.from_square)
            mt = mirror_square(move.to_square)
            manual = encode_move(chess.Move(mf, mt,
                                            promotion=move.promotion))
            if mirror_action_index(idx) != manual:
                print(f"Mirror mismatch: {move.uci()} {idx}")
                errors += 1
                break

    print(f"Move-encoding self-test: {checked} moves checked, {errors} errors.")
    return errors == 0


if __name__ == "__main__":
    _self_test()

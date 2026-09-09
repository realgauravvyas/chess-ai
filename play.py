"""Play chess against the trained model from the command line."""
import argparse
import sys
from pathlib import Path

import chess
import torch

from config import Config
from model import AlphaZeroNet
from mcts import MCTS
from utils import load_checkpoint


def default_checkpoint():
    """Best available model: a gated winner, else the newest checkpoint.

    The old default was the literal string "checkpoints/latest.pt" - a
    relative path that breaks outside the project root, pointing at the v5
    run's final weights, which measured 31.2% against the pretrained
    baseline. Iteration numbers are not comparable across runs, so ordering
    is by modification time.
    """
    root = Path(__file__).resolve().parent
    dirs = [root / "checkpoints"]
    dirs += sorted((root / "runs").glob("*/checkpoints"))
    for pattern in ("best.pt", "iter_*.pt", "latest.pt"):
        found = [q for d in dirs for q in d.glob(pattern)]
        if found:
            return max(found, key=lambda q: q.stat().st_mtime)
    return None

WHITE_SYMBOLS = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
                 chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
BLACK_SYMBOLS = {chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b",
                 chess.ROOK: "r", chess.QUEEN: "q", chess.KING: "k"}


def render(board, ascii_mode=False):
    if ascii_mode:
        white, black = WHITE_SYMBOLS, BLACK_SYMBOLS
    else:
        white = {chess.PAWN: "\u2659", chess.KNIGHT: "\u2658", chess.BISHOP: "\u2657",
                 chess.ROOK: "\u2656", chess.QUEEN: "\u2655", chess.KING: "\u2654"}
        black = {chess.PAWN: "\u265F", chess.KNIGHT: "\u265E", chess.BISHOP: "\u265D",
                 chess.ROOK: "\u265C", chess.QUEEN: "\u265B", chess.KING: "\u265A"}

    print("  a b c d e f g h")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                row += ". "
            else:
                symbols = white if piece.color == chess.WHITE else black
                row += symbols[piece.piece_type] + " "
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Play chess against the trained model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint to play against "
                             "(default: best available)")
    parser.add_argument("--sims", type=int, default=400,
                        help="MCTS simulations per model move.")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                        help="Color you want to play as.")
    parser.add_argument("--ascii", action="store_true", help="Use ASCII pieces.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to run the model on (default: cpu).")
    args = parser.parse_args()

    cfg = Config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = args.checkpoint or default_checkpoint()
    if ckpt is None:
        sys.exit("no checkpoint found; train one first or pass --checkpoint")
    args.checkpoint = str(ckpt)

    net = AlphaZeroNet(cfg.planes, cfg.filters, cfg.res_blocks,
                       action_planes=cfg.policy_size // 64).to(device)
    blob = load_checkpoint(args.checkpoint, map_location=device)
    net.load_state_dict(blob["model_state_dict"])
    net.eval()
    print(f"Loaded {args.checkpoint} | device {device}")

    board = chess.Board()
    mcts = MCTS(net, device, cfg.c_puct, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    human_white = args.color == "white"

    while not board.is_game_over():
        render(board, args.ascii)
        human_turn = (board.turn == chess.WHITE) == human_white

        if human_turn:
            cmd = input("Your move (UCI, e.g. e2e4; 'quit' to exit): ").strip().lower()
            if cmd in ("quit", "exit", "q"):
                sys.exit(0)
            if cmd == "moves":
                print(" ".join(m.uci() for m in board.legal_moves))
                continue
            try:
                move = chess.Move.from_uci(cmd)
            except ValueError:
                print("Invalid move format. Use UCI like 'e2e4' or 'g1f3'.")
                continue
            # Promotion is mandatory, so "e7e8" parses but is never legal.
            # Auto-queen rather than leaving the player stuck.
            if (move not in board.legal_moves and move.promotion is None
                    and board.piece_type_at(move.from_square) == chess.PAWN
                    and chess.square_rank(move.to_square) in (0, 7)):
                queened = chess.Move(move.from_square, move.to_square,
                                     promotion=chess.QUEEN)
                if queened in board.legal_moves:
                    move = queened
                    print(f"Promoting to a queen ({move.uci()}). "
                          f"For a knight, type {cmd}n.")
            if move not in board.legal_moves:
                print("Illegal move, try again.")
                continue
        else:
            print("Model is thinking...")
            probs = mcts.get_action_probs(board, args.sims)
            if not probs:
                break
            move = max(probs, key=probs.get)
            print(f"Model plays: {move.uci()}")

        board.push(move)
        print()

    render(board, args.ascii)
    print("Game over:", board.result())


if __name__ == "__main__":
    main()

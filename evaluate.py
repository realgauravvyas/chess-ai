"""Baseline opponents and evaluation harness for the trained network."""
import random

import chess
import numpy as np
import torch

from config import Config
from mcts import MCTS
from move_encoding import encode_move
from utils import board_to_planes


@torch.no_grad()
def greedy_policy_move(net, device, board):
    """Pick the legal move with the highest policy prior (no search)."""
    x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(device)
    logits, _ = net(x)
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    legal = list(board.legal_moves)
    return max(legal, key=lambda m: probs[encode_move(m)])


def play_eval_game(net, device, cfg: Config, opponent="random", num_sims=None,
                   net_plays_white=True, random_plies=0):
    """Play one game: network (with MCTS) vs a baseline. Returns result string.

    `random_plies` plays that many uniformly random opening plies before the
    real game starts. Without it, two deterministic players replay the exact
    same game every time and an N-game match carries only one game of
    information.
    """
    board = chess.Board()
    for _ in range(random_plies):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    mcts = MCTS(net, device, cfg.c_puct, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    sims = num_sims or cfg.num_simulations
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < cfg.max_game_length:
        net_to_move = (board.turn == chess.WHITE) == net_plays_white
        if net_to_move:
            probs = mcts.get_action_probs(board, sims)
            if not probs:
                break
            move = max(probs, key=probs.get)
        elif opponent == "random":
            move = random.choice(list(board.legal_moves))
        elif opponent == "greedy":
            move = greedy_policy_move(net, device, board)
        else:
            raise ValueError(f"Unknown opponent: {opponent}")

        board.push(move)
        plies += 1

    return board.result(claim_draw=True) if board.is_game_over(claim_draw=True)         else "1/2-1/2"


def net_points(result, net_plays_white):
    """Score a PGN result string from the NETWORK's point of view.

    The network alternates colors during evaluation, so a raw "1-0"/"0-1"
    count is white's score, not the network's. Everything that scores an
    evaluation must go through here.
    """
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == net_plays_white else 0.0


def evaluate_vs_baseline(net, device, cfg: Config, opponent="random",
                         num_games=20, num_sims=None, random_plies=2):
    """Play `num_games` games with the network as both colors vs a baseline.

    Returns win/draw/loss counts and the score fraction, all from the
    network's perspective.
    """
    tally = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(max(1, num_games // 2)):
        for net_white in (True, False):
            result = play_eval_game(net, device, cfg, opponent, num_sims,
                                    net_plays_white=net_white,
                                    random_plies=random_plies)
            pts = net_points(result, net_white)
            tally["win" if pts == 1.0 else "draw" if pts == 0.5 else "loss"] += 1
    played = sum(tally.values())
    tally["played"] = played
    tally["score"] = (tally["win"] + 0.5 * tally["draw"]) / played
    return tally


def match_nets(net_a, net_b, device, cfg: Config, num_games=12, num_sims=100,
               random_plies=4, seed=0):
    """Head-to-head match between two networks. Returns A's score fraction.

    Colors alternate and each game opens with `random_plies` random moves:
    two deterministic searchers otherwise replay one identical game, so an
    N-game match would carry only a single game of information.
    """
    import chess as _chess
    rng = random.Random(seed)
    total = 0.0
    for g in range(num_games):
        a_white = (g % 2 == 0)
        board = _chess.Board()
        for _ in range(random_plies):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        searchers = {
            _chess.WHITE: MCTS(net_a if a_white else net_b, device, cfg.c_puct,
                               dirichlet_alpha=0.0, dirichlet_epsilon=0.0),
            _chess.BLACK: MCTS(net_b if a_white else net_a, device, cfg.c_puct,
                               dirichlet_alpha=0.0, dirichlet_epsilon=0.0),
        }
        while not board.is_game_over(claim_draw=True) and board.ply() < cfg.max_game_length:
            probs = searchers[board.turn].get_action_probs(board, num_sims)
            if not probs:
                break
            board.push(max(probs, key=probs.get))
        over = board.is_game_over(claim_draw=True)
        result = board.result(claim_draw=True) if over else "1/2-1/2"
        total += net_points(result, a_white)
    return total / num_games

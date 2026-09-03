"""Monte Carlo Tree Search with a policy+value network (AlphaZero-style).

Node values are stored from the perspective of the side to move at that node.
"""
import math

import chess
import numpy as np
import torch

from move_encoding import encode_move
from utils import board_to_planes


class Node:
    __slots__ = ("prior", "visit_count", "total_value", "parent", "move",
                 "children", "is_terminal", "terminal_value")

    def __init__(self, prior=0.0, parent=None, move=None):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.parent = parent
        self.move = move
        self.children = {}
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def value(self):
        return self.total_value / (self.visit_count + 1e-8)


class MCTS:
    # Plies of history carried into each simulation so that repetitions are
    # visible to the search. Copying the full stack every simulation is
    # expensive; 12 plies covers the shuffling repetitions seen in practice.
    REPETITION_HISTORY = 12

    def __init__(self, net, device, c_puct=1.5, dirichlet_alpha=0.3,
                 dirichlet_epsilon=0.25, repetition_aware=False):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # When False the search cannot see repetitions at all, which is how
        # the published experiments were run. Measured over 20 games at 60
        # sims, enabling it does NOT improve playing strength (42.5%, p~0.5)
        # -- but it eliminates threefold repetitions (4/6 games -> 0/20) and
        # raises the decisive-game rate from ~33% to ~55%, which makes
        # head-to-head matches far more informative per game. Useful for
        # strengthening the acceptance gate, not for strengthening play.
        self.repetition_aware = repetition_aware

    @torch.no_grad()
    def _evaluate(self, board):
        self.net.eval()
        dev = next(self.net.parameters()).device  # follow the net's device
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(dev)
        logits, value = self.net(x)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        return probs, float(value.item())

    def _run_simulation(self, board):
        board = board.copy(
            stack=self.REPETITION_HISTORY if self.repetition_aware else False)
        node = self.root
        path = [node]

        while node.children:
            move = self._select(node)
            board.push(move)
            node = node.children[move]
            path.append(node)

        # A position repeated inside the search is a draw the side to move
        # can force, so score it as one. Without this the engine shuffles
        # into threefold repetitions it never saw coming.
        if self.repetition_aware and board.is_repetition(2):
            value = 0.0
            node.is_terminal = True
            node.terminal_value = 0.0
        elif node.is_terminal:
            value = node.terminal_value
        elif board.is_game_over():
            value = self._terminal_value(board)
            node.is_terminal = True
            node.terminal_value = value
        else:
            probs, value = self._evaluate(board)
            self._expand(node, board, probs)

        sign = 1
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += sign * value
            sign = -sign

    def _terminal_value(self, board):
        result = board.result()
        if result == "1/2-1/2":
            return 0.0
        white_wins = result == "1-0"
        if board.turn == chess.WHITE:
            return 1.0 if white_wins else -1.0
        return -1.0 if white_wins else 1.0

    def _select(self, node):
        best_move = None
        best_score = -math.inf
        sqrt_visits = math.sqrt(node.visit_count)
        for move, child in node.children.items():
            score = (-child.value
                     + self.c_puct * child.prior * sqrt_visits / (1 + child.visit_count))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _expand(self, node, board, probs):
        for move in board.legal_moves:
            idx = encode_move(move)
            node.children[move] = Node(prior=float(probs[idx]), parent=node, move=move)

        if node is self.root and self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
            eps = self.dirichlet_epsilon
            for child, n in zip(node.children.values(), noise):
                child.prior = (1 - eps) * child.prior + eps * n

    def get_action_probs(self, board, num_simulations=100):
        """Run MCTS from `board` and return the full normalized visit
        distribution {move: probability}."""
        self.root = Node()
        self.last_visits = {}
        self.root_value = 0.0
        for _ in range(num_simulations):
            self._run_simulation(board)

        visits = {m: c.visit_count for m, c in self.root.children.items()}
        if not visits:
            return {}
        self.last_visits = dict(visits)
        self.root_value = self.root.total_value / max(1, self.root.visit_count)
        total = sum(visits.values())
        return {m: c / total for m, c in visits.items()}

    def pv_line(self, board, max_len=8):
        """Principal variation (SAN) by greedily following visit counts."""
        b = board.copy(stack=False)
        node = self.root
        line = []
        while node.children and len(line) < max_len:
            mv, child = max(node.children.items(),
                            key=lambda kv: kv[1].visit_count)
            try:
                line.append(b.san(mv))
            except Exception:  # noqa: BLE001
                break
            b.push(mv)
            node = child
        return line

import random
import math
from typing import Tuple, Optional, List, Callable

ROCK = 'R'
PAPER = 'P'
SCISSORS = 'S'
MOVES = [ROCK, PAPER, SCISSORS]
MOVES_NAMES = {ROCK: "Камінь", PAPER: "Папір", SCISSORS: "Ножиці"}
MAX_SCORE = 7

def get_round_winner(move_m: str, move_o: str) -> str:
    if move_m == move_o:
        return "draw"
    if (move_m == ROCK and move_o == SCISSORS) or \
       (move_m == SCISSORS and move_o == PAPER) or \
       (move_m == PAPER and move_o == ROCK):
        return "max"
    return "min"

def get_next_score(score_m: int, score_o: int, move_m: str, move_o: str) -> Tuple[int, int]:
    winner = get_round_winner(move_m, move_o)
    if winner == "max":
        return score_m + 1, score_o
    elif winner == "min":
        return score_m, score_o + 1
    return score_m, score_o

def evaluate_state(score_m: int, score_o: int) -> float:
    if score_m >= MAX_SCORE: return float('inf')
    if score_o >= MAX_SCORE: return float('-inf')
    
    return score_m - score_o

TieBreakerFunc = Callable[[List[str]], str]

def random_tie_breaker(optimal_moves: List[str]) -> str:
    return random.choice(optimal_moves) if optimal_moves else random.choice(MOVES)

class MoveResult:
    def __init__(self, move: str, value: float, nodes: int):
        self.move = move
        self.value = value
        self.nodes = nodes

def minimax(score_m: int, score_o: int, depth: int, tie_breaker: TieBreakerFunc = random_tie_breaker) -> MoveResult:
    nodes = 0
    
    def recurse(d: int, is_max_turn: bool, sm: int, so: int) -> Tuple[Optional[str], float]:
        nonlocal nodes
        nodes += 1
        if d == 0 or sm >= MAX_SCORE or so >= MAX_SCORE:
            return None, evaluate_state(sm, so)

        if is_max_turn:
            best_val = -math.inf
            optimal_moves = []
            for move_m in MOVES:
                min_val_for_move = math.inf
                for move_o in MOVES:
                    nsm, nso = get_next_score(sm, so, move_m, move_o)
                    _, val = recurse(d - 1, False, nsm, nso)
                    min_val_for_move = min(min_val_for_move, val)

                if min_val_for_move > best_val:
                    best_val, optimal_moves = min_val_for_move, [move_m]
                elif min_val_for_move == best_val:
                    optimal_moves.append(move_m)
            return tie_breaker(optimal_moves), best_val
        else:
            best_val = math.inf
            for move_o in MOVES:
                max_val_for_move = -math.inf
                for move_m in MOVES:
                    nsm, nso = get_next_score(sm, so, move_m, move_o)
                    _, val = recurse(d - 1, True, nsm, nso)
                    max_val_for_move = max(max_val_for_move, val)
                best_val = min(best_val, max_val_for_move)
            return None, best_val

    best_move, val = recurse(depth, True, score_m, score_o)
    return MoveResult(best_move if best_move else random.choice(MOVES), val, nodes)

def play_series(agent_depth=3) -> bool :
    score_m, score_o = 0, 0
    round_num = 1

    print(f"=== ПОЧАТОК СЕРІЇ RPS ДО {MAX_SCORE} ПЕРЕМОГ ===")
    print(f"Агент: MiniMax (глибина {agent_depth}) vs Противник: Random\n")

    while score_m < MAX_SCORE and score_o < MAX_SCORE:
        result = minimax(score_m, score_o, agent_depth, random_tie_breaker)
        agent_move = result.move
        opponent_move = random.choice(MOVES)

        winner = get_round_winner(agent_move, opponent_move)
        score_m, score_o = get_next_score(score_m, score_o, agent_move, opponent_move)

        winner_text = "НІЧИЯ"
        if winner == "max": winner_text = "Перемога Агента"
        elif winner == "min": winner_text = "Перемога Противника"

        print(f"--- Раунд {round_num} ---")
        print(f"Агент: {MOVES_NAMES[agent_move]} | Противник: {MOVES_NAMES[opponent_move]}")
        print(f"Результат: {winner_text} | Рахунок: {score_m}-{score_o}")
        print("-" * 30)
        round_num += 1

    print("\n=== СЕРІЮ ЗАВЕРШЕНО ===")
    if score_m >= MAX_SCORE:
        print(f"Агент MiniMax ПЕРЕМІГ з рахунком {score_m}:{score_o}!")
        return True
    else:
        print(f"Агент MiniMax ПРОГРАВ з рахунком {score_m}:{score_o}.")
        return False

if __name__ == "__main__":
    results = []

    for _ in range(300):
        res = play_series(agent_depth=4)
        results.append(res)

    per = (results.count(True) / results.__len__()) * 100
    print(f"{per}% of wins")
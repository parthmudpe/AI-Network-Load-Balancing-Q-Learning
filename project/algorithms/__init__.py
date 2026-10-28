"""
Every algorithm is a small class with a  select(servers)  method that returns
the server that should receive the next request (or None if none is possible).
"""
from algorithms.round_robin import RoundRobin
from algorithms.weighted_round_robin import WeightedRoundRobin
from algorithms.least_connections import LeastConnections
from algorithms.random_selection import RandomSelection
from algorithms.multi_factor import MultiFactor, DEFAULT_WEIGHTS
from algorithms.q_learning import QLearning, train as train_q_learning

ALGORITHM_NAMES = [
    "Round Robin",
    "Weighted Round Robin",
    "Least Connections",
    "Random Selection",
    "Multi-Factor Dynamic Selection",
    "Q-Learning (AI)",
]

AI_NAME = "Q-Learning (AI)"      # the only algorithm that has to be trained first

# short names for chart labels
SHORT_NAMES = {
    "Round Robin": "Round\nRobin",
    "Weighted Round Robin": "Weighted\nRR",
    "Least Connections": "Least\nConn.",
    "Random Selection": "Random",
    "Multi-Factor Dynamic Selection": "Multi-\nFactor",
    "Q-Learning (AI)": "Q-Learning\n(AI)",
}


def make_algorithm(name, weights=None, seed=1, q_table=None):
    """
    Create a fresh algorithm object (fresh = no leftover state from earlier runs).
    q_table = what the Q-Learning agent learned during training (only used for Q-Learning).
    The agent made here only USES the table (epsilon = 0, learning off).
    """
    if name == "Round Robin":
        return RoundRobin()
    if name == "Weighted Round Robin":
        return WeightedRoundRobin()
    if name == "Least Connections":
        return LeastConnections()
    if name == "Random Selection":
        return RandomSelection(seed)
    if name == "Multi-Factor Dynamic Selection":
        return MultiFactor(weights)
    if name == "Q-Learning (AI)":
        return QLearning(q_table, learning=False, epsilon=0.0, seed=seed)
    raise ValueError("Unknown algorithm: " + str(name))

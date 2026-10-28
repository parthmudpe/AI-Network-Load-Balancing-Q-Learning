"""Random Selection: pick any available server at random."""
import numpy as np


class RandomSelection:
    name = "Random Selection"

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)     # seed -> repeatable results

    def select(self, servers):
        available = [s for s in servers if s.available]
        if not available:
            return None
        return available[int(self.rng.integers(len(available)))]

"""
Weighted Round Robin: a server with a higher capacity gets more turns.

Weight = capacity / 10   (100 req/s -> 10, 150 req/s -> 15, ...)

Building the schedule (example weights S1=2, S2=3, S3=1):
    Level 1: every server with weight >= 1  ->  S1 S2 S3
    Level 2: every server with weight >= 2  ->  S1 S2
    Level 3: every server with weight >= 3  ->  S2
    Schedule = S1 S2 S3 S1 S2 S2   (S1 gets 2 turns, S2 gets 3, S3 gets 1)
The schedule is repeated again and again.
"""


class WeightedRoundRobin:
    name = "Weighted Round Robin"

    def __init__(self):
        self.counter = 0                 # position in the schedule
        self._schedule_key = None        # which servers the saved schedule was built for
        self._schedule = []

    def select(self, servers):
        available = [s for s in servers if s.available]     # skip FAILED servers
        if not available:
            return None

        # build one full cycle of the schedule (only again if a server failed/recovered,
        # because with 25 servers the schedule can be over a thousand entries long)
        key = tuple(s.index for s in available)
        if key != self._schedule_key:
            highest_weight = max(s.weight for s in available)
            schedule = []
            for level in range(1, highest_weight + 1):
                for s in available:
                    if s.weight >= level:
                        schedule.append(s)
            self._schedule_key = key
            self._schedule = schedule

        schedule = self._schedule
        chosen = schedule[self.counter % len(schedule)]
        self.counter += 1
        return chosen

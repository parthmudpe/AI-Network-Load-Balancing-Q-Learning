"""
server.py - A simulated server.

How a server works (simple model):
  * A server has some number of "slots" (workers). Each slot handles ONE
    request at a time.           slots = capacity / 10
  * If all slots are busy, new requests wait in a FIFO queue.
  * If the queue is also full, the request is DROPPED.
  * Together, all slots can process `capacity` requests per second.
"""
from dataclasses import dataclass

from simulation.request import Request


@dataclass
class ServerConfig:
    """Settings the user can change in the Configuration tab."""
    name: str
    capacity: int = 100        # requests per second
    max_queue: int = 20        # maximum waiting requests
    initial_load: int = 0      # % of slots already busy when the simulation starts
    response_time: int = 50    # base response time in ms
    availability: int = 99     # reliability rating in % (used by Multi-Factor)
    up: bool = True            # True = AVAILABLE, False = FAILED at the start


def default_configs():
    """Four servers with different values so the algorithms behave differently."""
    return [
        ServerConfig("Server 1", capacity=100, max_queue=20, initial_load=10, response_time=50, availability=99),
        ServerConfig("Server 2", capacity=150, max_queue=30, initial_load=0,  response_time=70, availability=98),
        ServerConfig("Server 3", capacity=80,  max_queue=15, initial_load=20, response_time=40, availability=95),
        ServerConfig("Server 4", capacity=120, max_queue=25, initial_load=5,  response_time=60, availability=99),
    ]


class Server:
    def __init__(self, config, index, rng):
        self.index = index
        self.name = config.name
        self.capacity = config.capacity
        self.max_queue = config.max_queue
        self.response_time = config.response_time
        self.availability = config.availability
        self.available = config.up

        # slots = number of requests processed in parallel
        self.slots = max(1, round(config.capacity / 10))
        # each slot works at this speed so that all slots together = capacity
        self.slot_speed = config.capacity / self.slots

        self.active = []          # requests being processed right now
        self.queue = []           # requests waiting
        self.last_score = None    # last Multi-Factor score (only for display)

        # Initial load: some slots are already busy with older requests
        # (a server that starts FAILED has nothing running)
        busy_slots = round(config.initial_load / 100 * self.slots) if config.up else 0
        for _ in range(busy_slots):
            old_request = Request(0, 0.0, float(rng.uniform(2, 6)), is_background=True)
            self.active.append(old_request)

    # ---------- values used by the algorithms and the GUI ----------
    @property
    def weight(self):
        """Weight for Weighted Round Robin (capacity 100 -> weight 10)."""
        return self.slots

    @property
    def load_percent(self):
        """Current load = % of slots that are busy."""
        return 100.0 * len(self.active) / self.slots

    @property
    def connections(self):
        """Active connections = requests being processed + requests waiting."""
        return len(self.active) + len(self.queue)

    @property
    def is_full(self):
        """True when every slot is busy AND the queue is full."""
        return len(self.active) >= self.slots and len(self.queue) >= self.max_queue

    # ---------- actions ----------
    def accept(self, request, now):
        """Try to take a request. Returns False if the request must be dropped."""
        if len(self.active) < self.slots:
            request.start_time = now
            self.active.append(request)
            return True
        if len(self.queue) < self.max_queue:
            self.queue.append(request)
            return True
        return False                       # queue full -> dropped

    def advance(self, now, dt):
        """Move time forward by dt seconds. Returns the requests that finished."""
        finished = []
        still_active = []
        for request in self.active:
            request.remaining -= self.slot_speed * dt
            if request.remaining <= 0:
                request.finish_time = now + dt
                finished.append(request)
            else:
                still_active.append(request)
        self.active = still_active

        # free slots take requests from the front of the queue (FIFO)
        while self.queue and len(self.active) < self.slots:
            next_request = self.queue.pop(0)
            next_request.start_time = now
            self.active.append(next_request)

        return [r for r in finished if not r.is_background]

    def fail(self):
        """Server crashes. Everything it was handling is lost (counted as dropped)."""
        lost = [r for r in self.active + self.queue if not r.is_background]
        self.active = []
        self.queue = []
        self.available = False
        return len(lost)

    def recover(self):
        self.available = True

"""
simulator.py - The simulation engine.

The simulation moves forward in small time steps (DT = 0.01 s).
In every step:
    1. Requests whose arrival time has come reach the load balancer.
       The chosen algorithm picks a server for each of them.
       (If the server's queue is full, the request is DROPPED.)
    2. We measure utilization and queue lengths.
    3. Every server processes its requests for DT seconds.
The simulation is finished when all requests are done.
"""
import numpy as np

from simulation.server import Server
from simulation.metrics import Metrics


class Simulator:
    DT = 0.01                  # size of one time step in seconds
    MAX_EXTRA_TIME = 120.0     # safety limit: stop if it runs far too long

    def __init__(self, configs, requests, algorithm, duration, seed=1):
        rng = np.random.default_rng(seed)
        self.servers = [Server(c, i, rng) for i, c in enumerate(configs)]
        self.requests = requests
        self.algorithm = algorithm
        self.duration = duration

        self.time = 0.0
        self.tick = 0
        self.next_request = 0          # index of the next request to arrive
        self.finished = False
        self.recent = set()            # servers that got a request in the last step
        # Optional hooks: only the Q-Learning algorithm has them (it learns from outcomes).
        self._on_dispatch = getattr(algorithm, "observe_dispatch", None)
        self._on_completion = getattr(algorithm, "observe_completion", None)
        self.metrics = Metrics([s.name for s in self.servers])

    # ---------- server failure / recovery ----------
    def fail_server(self, index):
        lost = self.servers[index].fail()
        self.metrics.record_drop(lost)       # requests inside a crashed server are lost

    def recover_server(self, index):
        self.servers[index].recover()

    # ---------- main loop ----------
    def step(self):
        if self.finished:
            return
        self.recent.clear()

        # 1. new arrivals
        while (self.next_request < len(self.requests) and
               self.requests[self.next_request].arrival_time <= self.time):
            self._dispatch(self.requests[self.next_request])
            self.next_request += 1

        # 2. measurements
        self.metrics.sample(self.time, self.tick, self.servers)

        # 3. servers work for DT seconds
        for server in self.servers:
            for request in server.advance(self.time, self.DT):
                response_ms = (request.finish_time - request.arrival_time) * 1000 \
                              + server.response_time
                self.metrics.record_completion(server.index, response_ms)
                if self._on_completion is not None:
                    self._on_completion(request, server, response_ms)

        self.time += self.DT
        self.tick += 1

        # 4. done?
        all_arrived = self.next_request >= len(self.requests)
        all_idle = all(not s.active and not s.queue for s in self.servers)
        if (all_arrived and all_idle) or self.time > self.duration + self.MAX_EXTRA_TIME:
            self.finished = True

    def _dispatch(self, request):
        """Give one request to the server chosen by the algorithm."""
        self.metrics.record_arrival()
        server = self.algorithm.select(self.servers)
        if server is None:                        # no server available -> dropped
            self.metrics.record_drop()
            return
        self.metrics.record_assigned(server.index)
        self.recent.add(server.index)
        accepted = server.accept(request, self.time)
        if not accepted:
            self.metrics.record_drop()            # queue full -> dropped
        if self._on_dispatch is not None:
            self._on_dispatch(request, server, accepted)

    def run_to_end(self):
        """Run without the GUI (used by comparison mode and tests)."""
        while not self.finished:
            self.step()

    def summary(self):
        return self.metrics.summary(self.time)

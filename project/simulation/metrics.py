"""
metrics.py - Collects REAL measurements while the simulation runs.

Nothing here is fake: every number comes from what the simulator did.
"""
import numpy as np


class Metrics:
    def __init__(self, server_names):
        n = len(server_names)
        self.server_names = list(server_names)
        self.total_requests = 0
        self.dropped = 0

        # per-server counters
        self.assigned = np.zeros(n)          # requests sent to each server
        self.handled = np.zeros(n)           # requests each server finished
        self.response_sum = np.zeros(n)      # sum of response times (ms)
        self.response_times = []             # every response time (ms)

        # sampled every step while the simulation runs
        self.samples = 0
        self.util_sum = np.zeros(n)          # sum of load fractions
        self.queue_sum = np.zeros(n)         # sum of queue lengths

        # queue length history for the line chart (every 0.1 s)
        self.history_time = []
        self.history_queue = []

    # ---------- recording ----------
    def record_arrival(self):
        self.total_requests += 1

    def record_assigned(self, server_index):
        self.assigned[server_index] += 1

    def record_drop(self, count=1):
        self.dropped += count

    def record_completion(self, server_index, response_ms):
        self.handled[server_index] += 1
        self.response_sum[server_index] += response_ms
        self.response_times.append(response_ms)

    def sample(self, now, tick, servers):
        """Called once per simulation step to measure utilization and queues."""
        self.samples += 1
        for i, s in enumerate(servers):
            self.util_sum[i] += len(s.active) / s.slots
            self.queue_sum[i] += len(s.queue)
        if tick % 10 == 0:
            self.history_time.append(now)
            self.history_queue.append([len(s.queue) for s in servers])

    # ---------- results ----------
    def summary(self, elapsed_time):
        """Return all metrics as a dictionary."""
        processed = int(self.handled.sum())
        samples = max(self.samples, 1)

        utilization = 100.0 * self.util_sum / samples          # % per server
        queue_avg = self.queue_sum / samples                   # per server
        assigned_total = self.assigned.sum()

        server_response = np.divide(self.response_sum, self.handled,
                                    out=np.zeros_like(self.response_sum),
                                    where=self.handled > 0)

        return {
            "names": self.server_names,
            "total": self.total_requests,
            "processed": processed,
            "dropped": self.dropped,
            "avg_response": float(np.mean(self.response_times)) if self.response_times else 0.0,
            "throughput": processed / elapsed_time if elapsed_time > 0 else 0.0,
            "avg_utilization": float(utilization.mean()),
            "avg_queue": float(queue_avg.mean()),
            # per server
            "handled": self.handled.tolist(),
            "assigned": self.assigned.tolist(),
            "utilization": utilization.tolist(),
            "server_response": server_response.tolist(),
            "queue_avg": queue_avg.tolist(),
            "distribution": (100.0 * self.assigned / assigned_total).tolist()
                            if assigned_total > 0 else [0.0] * len(self.server_names),
            # history for the queue chart
            "history_time": list(self.history_time),
            "history_queue": [list(row) for row in self.history_queue],
        }

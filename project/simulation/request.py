"""
request.py - What a client request is, and how traffic is generated.

A request has only three important properties (as required by the project):
    1. Request ID
    2. Arrival time        (in seconds from the start of the simulation)
    3. Processing requirement ("work" - 1.0 means an average request)

We do NOT simulate real network packets - a request is just a small object.
"""
import numpy as np

TRAFFIC_MODES = ["Normal", "High", "Burst"]


class Request:
    def __init__(self, request_id, arrival_time, work, is_background=False):
        self.request_id = request_id
        self.arrival_time = arrival_time      # when the request reaches the load balancer
        self.work = work                      # processing requirement
        self.remaining = work                 # work still left to do
        self.start_time = None                # when a server started processing it
        self.finish_time = None               # when a server finished it
        # Background requests represent the "initial load" a server already has
        # when the simulation starts. They are NOT counted in the metrics.
        self.is_background = is_background


def generate_requests(mode, num_requests, duration, seed=42):
    """
    Create the list of requests for one simulation.

    The same (mode, num_requests, duration, seed) always gives the SAME list.
    This is what makes the algorithm comparison fair.

    NORMAL : requests arrive at a low, steady rate over the whole duration.
    HIGH   : the same number of requests arrive in half the time (2x rate).
    BURST  : 25% arrive steadily, but 75% arrive suddenly in a short window
             (10% of the duration, starting at 40% of the duration).
    """
    rng = np.random.default_rng(seed)

    if mode == "Normal":
        arrivals = _steady_arrivals(num_requests, duration, rng)
    elif mode == "High":
        arrivals = _steady_arrivals(num_requests, duration * 0.5, rng)
    elif mode == "Burst":
        num_burst = int(num_requests * 0.75)
        num_steady = num_requests - num_burst
        steady = _steady_arrivals(num_steady, duration, rng)
        burst_start = duration * 0.4
        burst_length = duration * 0.1
        burst = burst_start + rng.uniform(0, burst_length, num_burst)
        arrivals = np.sort(np.concatenate([steady, burst]))
    else:
        raise ValueError("Unknown traffic mode: " + str(mode))

    # Processing requirement: random between 0.5 and 1.5 (average 1.0)
    work_values = rng.uniform(0.5, 1.5, num_requests)

    return [Request(i + 1, float(arrivals[i]), float(work_values[i]))
            for i in range(num_requests)]


def _steady_arrivals(count, window, rng):
    """Evenly spaced arrival times over 'window' seconds, with a little jitter."""
    if count <= 0:
        return np.array([])
    gap = window / count
    times = np.arange(count) * gap + rng.uniform(0, gap, count)
    return np.sort(times)

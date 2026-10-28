"""
Q-Learning (AI) load balancer  -  a SIMPLE, tabular reinforcement-learning method.
(No neural network, no deep learning: just a small table of numbers.)

IDEA
    For every incoming request the balancer looks at each available server,
    describes that server's current condition with a few small numbers (its
    "state"), looks the state up in the Q-table, and sends the request to the
    server whose state has the best Q-value.

    Because the state describes ONE server (not the whole system), the table
    stays tiny (5 x 5 x 3 x 2 = 150 states) and works for any number of
    servers, from 3 up to 25.

STATE of a server  (all values are small integers)
    load level     0..4   how many of its slots are busy      (4 = all busy)
    queue level    0..4   how full its waiting queue is       (4 = full)
    speed level    0..2   its capacity compared to the biggest server
    latency level  0..1   its base response time (0 = fast, 1 = slow)

ACTION
    "Send this request to that server."

REWARD  (learned from what really happens to the request)
    request finished   ->  reward = - (its response time in seconds)
    request dropped    ->  reward = - DROP_PENALTY
    A short response time is therefore good (reward close to 0).

LEARNING RULE  (standard Q-Learning update)
    Q(state) = Q(state) + ALPHA * ( reward + GAMMA * Q(next state) - Q(state) )
    "next state" = the same server's state just after it accepted the request.

TRAINING
    The agent learns by trial and error. During training it sometimes tries a
    random server (epsilon-greedy exploration) so it can discover which server
    conditions give short response times. After training, epsilon = 0 and the
    agent only uses what it has learned ("greedy"), and it stops learning, so
    the comparison with the other algorithms is fair and repeatable.
"""
import copy

import numpy as np

# ---------- learning settings (kept in one place so they are easy to explain) ----------
ALPHA = 0.1             # learning rate: how fast new experience replaces old knowledge
GAMMA = 0.5             # discount: how much the following state matters
DROP_PENALTY = 10.0     # reward is -10 when a request is dropped (very bad)

TRAIN_EPISODES = 25         # number of practice simulations
TRAIN_MAX_DURATION = 10.0   # practice simulations are at most this many seconds long
TRAIN_MAX_REQUESTS = 3000   # ... and at most this many requests (keeps training quick)
EPSILON_START = 0.30        # at first, 30% of the choices are random (exploration)
EPSILON_MIN = 0.05          # ... and this falls to 5% by the end of training
EPSILON_DECAY = 0.85        # epsilon is multiplied by this after every episode

NAME = "Q-Learning (AI)"


def server_state(server, max_capacity, min_response):
    """Describe one server's current condition with four small integers."""
    load = server.load_percent / 100.0
    load_level = min(4, int(load * 4))                       # 0,1,2,3 ... 4 = all slots busy

    queue_fill = len(server.queue) / server.max_queue
    if queue_fill <= 0:
        queue_level = 0                                      # empty
    elif queue_fill >= 1:
        queue_level = 4                                      # full
    else:
        queue_level = 1 + int(queue_fill * 3)                # 1, 2 or 3

    ratio = server.capacity / max_capacity
    speed_level = 0 if ratio < 0.5 else (1 if ratio < 0.8 else 2)

    latency_level = 0 if server.response_time <= 1.5 * min_response else 1
    return (load_level, queue_level, speed_level, latency_level)


def starting_value(state):
    """
    Q-value of a state the agent has never seen. A small penalty for busy
    servers, so that an untrained agent already prefers quiet servers.
    """
    return -0.05 * (state[0] + 2 * state[1])


class QLearning:
    name = NAME

    def __init__(self, q_table=None, learning=False, epsilon=0.0, seed=1):
        # q_table is a dictionary:  state -> Q-value.  It can be shared between runs.
        self.q_table = q_table if q_table is not None else {}
        self.learning = learning       # True only while training
        self.epsilon = epsilon         # chance of trying a random server
        self.rng = np.random.default_rng(seed)
        self._chosen_state = None      # state of the server picked by select()
        self._scale = (1, 1)           # (max capacity, min response) used for states
        self._pending = {}             # request id -> (state before, state after)

    def q_value(self, state):
        return self.q_table.get(state, starting_value(state))

    # ---------- choosing a server ----------
    def select(self, servers):
        for s in servers:
            s.last_score = None                      # only available servers get a Q-value
        available = [s for s in servers if s.available]
        if not available:
            return None

        self._scale = (max(s.capacity for s in servers), min(s.response_time for s in servers))
        states = [server_state(s, *self._scale) for s in available]
        values = [self.q_value(st) for st in states]
        for s, v in zip(available, values):
            s.last_score = v                         # shown on the server card

        if self.learning and self.rng.random() < self.epsilon:
            k = int(self.rng.integers(len(available)))        # explore: random server
        else:
            best = max(values)                                # exploit: best Q-value
            ties = [i for i, v in enumerate(values) if v >= best - 1e-9]
            k = ties[int(self.rng.integers(len(ties)))] if len(ties) > 1 else ties[0]

        self._chosen_state = states[k]
        return available[k]

    # ---------- learning (only used during training) ----------
    def observe_dispatch(self, request, server, accepted):
        """Called by the simulator right after the chosen server accepted/rejected the request."""
        if not self.learning:
            return
        if not accepted:                                      # dropped: learn immediately
            self._update(self._chosen_state, -DROP_PENALTY, None)
        else:                                                 # remember it until it finishes
            after = server_state(server, *self._scale)
            self._pending[request.request_id] = (self._chosen_state, after)

    def observe_completion(self, request, server, response_ms):
        """Called by the simulator when a request has finished; the reward is known now."""
        if not self.learning:
            return
        pair = self._pending.pop(request.request_id, None)
        if pair is not None:
            self._update(pair[0], -response_ms / 1000.0, pair[1])

    def _update(self, state, reward, next_state):
        old = self.q_value(state)
        future = self.q_value(next_state) if next_state is not None else 0.0
        self.q_table[state] = old + ALPHA * (reward + GAMMA * future - old)


def train(configs, mode, num_requests, duration, episodes=TRAIN_EPISODES, progress=None):
    """
    Let the agent practise on simulated traffic and return its Q-table.

    * Practice uses DIFFERENT random traffic (other seeds) than the final
      evaluation, so the AI is not simply memorising the test scenario.
    * Practice runs are shortened (at most TRAIN_MAX_DURATION seconds and
      TRAIN_MAX_REQUESTS requests) while keeping the traffic RATE the same.
    * progress(done, total) is called after every episode (used by the GUI).
    """
    # imported here to avoid a circular import (simulation does not import algorithms)
    from simulation.request import generate_requests
    from simulation.simulator import Simulator

    q_table = {}
    train_duration = min(float(duration), TRAIN_MAX_DURATION)
    train_requests = int(round(num_requests * train_duration / duration))
    if train_requests > TRAIN_MAX_REQUESTS:                  # shrink time too: same traffic rate
        train_duration *= TRAIN_MAX_REQUESTS / train_requests
        train_requests = TRAIN_MAX_REQUESTS
    train_requests = max(10, train_requests)
    epsilon = EPSILON_START
    for episode in range(episodes):
        requests = generate_requests(mode, train_requests, train_duration, seed=1000 + episode)
        agent = QLearning(q_table, learning=True, epsilon=epsilon, seed=episode + 1)
        sim = Simulator(copy.deepcopy(configs), requests, agent, train_duration, seed=episode + 1)
        sim.run_to_end()
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if progress is not None:
            progress(episode + 1, episodes)
    return q_table

"""
Multi-Factor Dynamic Server Selection  (our proposed method)

Every factor is converted to a number between 0 and 1 where
1 = BEST and 0 = WORST. Then we take a weighted sum:

    Score = w1 x Load + w2 x Queue + w3 x Response + w4 x Capacity + w5 x Availability

The server with the HIGHEST score gets the request.

Factor formulas (all easy to explain):
    Load factor         = 1 - (current load % / 100)          less load    -> better
    Queue factor        = 1 - (queue length / max queue)      shorter queue-> better
    Response factor     = fastest response time / this server's response time
    Capacity factor     = this server's capacity / highest capacity
    Availability factor = availability % / 100

Which servers are "valid"?  Only servers that are AVAILABLE (not failed)
and NOT completely full (all slots busy and queue full).
"""

DEFAULT_WEIGHTS = {
    "load": 0.25,
    "queue": 0.20,
    "response": 0.20,
    "capacity": 0.20,
    "availability": 0.15,
}


class MultiFactor:
    name = "Multi-Factor Dynamic Selection"

    def __init__(self, weights=None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def select(self, servers):
        valid = [s for s in servers if s.available and not s.is_full]

        for s in servers:
            s.last_score = None          # only valid servers get a score
        if not valid:
            return None

        fastest = min(s.response_time for s in valid)
        highest_capacity = max(s.capacity for s in valid)
        total_weight = sum(self.weights.values())      # so the score stays between 0 and 1

        best_server = None
        best_score = -1.0
        for s in valid:
            load_factor = 1 - s.load_percent / 100
            queue_factor = 1 - len(s.queue) / s.max_queue
            response_factor = fastest / s.response_time
            capacity_factor = s.capacity / highest_capacity
            availability_factor = s.availability / 100

            score = (self.weights["load"] * load_factor
                     + self.weights["queue"] * queue_factor
                     + self.weights["response"] * response_factor
                     + self.weights["capacity"] * capacity_factor
                     + self.weights["availability"] * availability_factor) / total_weight

            s.last_score = score
            if score > best_score:
                best_score = score
                best_server = s
        return best_server

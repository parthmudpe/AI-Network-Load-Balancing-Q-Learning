"""Round Robin: S1 -> S2 -> S3 -> S4 -> S1 -> ... (skips failed servers)."""


class RoundRobin:
    name = "Round Robin"

    def __init__(self):
        self.next_index = 0              # whose turn it is

    def select(self, servers):
        count = len(servers)
        for offset in range(count):
            i = (self.next_index + offset) % count
            if servers[i].available:                 # skip FAILED servers
                self.next_index = (i + 1) % count    # next turn goes to the next server
                return servers[i]
        return None                                  # every server has failed

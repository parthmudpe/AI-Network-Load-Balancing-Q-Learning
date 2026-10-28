"""Least Connections: send the request to the server with the fewest active connections."""


class LeastConnections:
    name = "Least Connections"

    def select(self, servers):
        available = [s for s in servers if s.available]
        if not available:
            return None
        # min() returns the first server if there is a tie
        return min(available, key=lambda s: s.connections)

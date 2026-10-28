"""
test_simulation.py - Basic checks for the simulation core.

Run with:   python -m tests.test_simulation
(from the project root)
"""
import copy

from simulation.server import ServerConfig, default_configs
from simulation.request import generate_requests
from simulation.simulator import Simulator
from algorithms.round_robin import RoundRobin
from algorithms.weighted_round_robin import WeightedRoundRobin
from algorithms.least_connections import LeastConnections
from algorithms.random_selection import RandomSelection
from algorithms.multi_factor import MultiFactor
from algorithms.q_learning import QLearning, server_state, train
from algorithms import ALGORITHM_NAMES, make_algorithm

ALGORITHMS = [RoundRobin(), WeightedRoundRobin(), LeastConnections(),
              RandomSelection(), MultiFactor()]


def run(mode, num_requests, duration, algo):
    configs = default_configs()
    requests = generate_requests(mode, num_requests, duration)
    sim = Simulator(configs, requests, algo, duration)
    sim.run_to_end()
    return sim


def test_all_algorithms_normal_traffic():
    for algo in ALGORITHMS:
        sim = run("Normal", 60, 10, algo)
        s = sim.summary()
        assert s["processed"] + s["dropped"] == s["total"]
        print(f"  {algo.__class__.__name__}: processed={s['processed']} "
              f"dropped={s['dropped']} avg_response={s['avg_response']:.1f}ms")


def test_high_traffic():
    sim = run("High", 100, 10, RoundRobin())
    s = sim.summary()
    assert s["processed"] + s["dropped"] == s["total"]


def test_burst_traffic():
    sim = run("Burst", 100, 10, LeastConnections())
    s = sim.summary()
    assert s["processed"] + s["dropped"] == s["total"]


def test_server_overload_drops_requests():
    # Tiny queues + huge burst of requests should force some drops.
    configs = default_configs()
    for c in configs:
        c.max_queue = 1
        c.capacity = 20
    requests = generate_requests("Burst", 300, 5)
    sim = Simulator(configs, requests, RoundRobin(), duration=5)
    sim.run_to_end()
    s = sim.summary()
    assert s["dropped"] > 0, "expected some drops under heavy overload"


def test_server_failure_and_recovery():
    sim = run("Normal", 40, 10, LeastConnections())
    sim.fail_server(0)
    assert sim.servers[0].available is False
    sim.recover_server(0)
    assert sim.servers[0].available is True


def test_failed_server_receives_no_new_requests():
    configs = default_configs()
    requests = generate_requests("Normal", 40, 10)
    sim = Simulator(configs, requests, RoundRobin(), duration=10)
    sim.fail_server(0)
    sim.run_to_end()
    s = sim.summary()
    # server 0 (index 0) should not have been assigned any new requests
    assert s["assigned"][0] == 0


def test_comparison_uses_same_scenario():
    # Same seed => identical request list for every algorithm, which is
    # what makes the Comparison tab a fair comparison.
    r1 = generate_requests("Normal", 50, 10)
    r2 = generate_requests("Normal", 50, 10)
    assert [req.arrival_time for req in r1] == [req.arrival_time for req in r2]


# ---------------------------------------------------------------------
# Q-Learning (AI), 25 servers, and fixes
# ---------------------------------------------------------------------
def many_servers(count):
    """`count` servers with mixed settings (for the 25-server tests)."""
    return [ServerConfig("S%d" % (i + 1), capacity=[80, 100, 120, 150][i % 4],
                         max_queue=[15, 20, 25, 30][i % 4], initial_load=[0, 10, 20, 5][i % 4],
                         response_time=[40, 50, 60, 70][i % 4], availability=[99, 98, 95, 99][i % 4])
            for i in range(count)]


def run_with(configs, mode, n, duration, algo, seed=42):
    sim = Simulator(copy.deepcopy(configs), generate_requests(mode, n, duration, seed), algo, duration)
    sim.run_to_end()
    return sim


def test_ai_is_in_the_algorithm_list():
    assert "Q-Learning (AI)" in ALGORITHM_NAMES
    assert len(ALGORITHM_NAMES) == 6
    assert make_algorithm("Q-Learning (AI)").name == "Q-Learning (AI)"


def test_ai_training_builds_a_q_table():
    q_table = train(default_configs(), "High", 600, 5, episodes=5)
    assert len(q_table) > 0
    assert all(isinstance(v, float) for v in q_table.values())
    # every state is four small integers
    for state in q_table:
        assert len(state) == 4 and state[0] in range(5) and state[1] in range(5)
        assert state[2] in range(3) and state[3] in range(2)


def test_ai_learning_changes_the_table_but_evaluation_does_not():
    q_table = train(default_configs(), "High", 600, 5, episodes=5)
    before = dict(q_table)
    algo = QLearning(q_table, learning=False, epsilon=0.0)       # evaluation: must NOT learn
    sim = run_with(default_configs(), "High", 600, 5, algo)
    assert q_table == before, "the Q-table must not change while the AI is only being evaluated"
    s = sim.summary()
    assert s["processed"] + s["dropped"] == s["total"]


def test_ai_is_repeatable():
    q_table = train(default_configs(), "Normal", 500, 5, episodes=4)
    a = run_with(default_configs(), "Normal", 500, 5, make_algorithm("Q-Learning (AI)", q_table=q_table)).summary()
    b = run_with(default_configs(), "Normal", 500, 5, make_algorithm("Q-Learning (AI)", q_table=q_table)).summary()
    assert a["avg_response"] == b["avg_response"] and a["handled"] == b["handled"]


def test_ai_never_picks_a_failed_server():
    q_table = train(default_configs(), "Normal", 400, 5, episodes=3)
    configs = default_configs()
    configs[1].up = False
    sim = run_with(configs, "Normal", 300, 5, make_algorithm("Q-Learning (AI)", q_table=q_table))
    assert sim.summary()["assigned"][1] == 0


def test_ai_with_all_servers_failed_drops_everything():
    configs = default_configs()
    for c in configs:
        c.up = False
    sim = run_with(configs, "Normal", 50, 2, QLearning())
    s = sim.summary()
    assert s["processed"] == 0 and s["dropped"] == 50


def test_ai_handles_untrained_table():
    # an untrained agent still works (it prefers quiet servers through its starting values)
    sim = run_with(default_configs(), "Normal", 200, 3, QLearning())
    s = sim.summary()
    assert s["processed"] + s["dropped"] == s["total"] and s["processed"] > 0


def test_server_state_values_are_in_range():
    sim = run_with(default_configs(), "High", 300, 3, RoundRobin())
    for server in sim.servers:
        state = server_state(server, 150, 40)
        assert 0 <= state[0] <= 4 and 0 <= state[1] <= 4 and 0 <= state[2] <= 2 and state[3] in (0, 1)


def test_25_servers_all_algorithms():
    configs = many_servers(25)
    q_table = train(configs, "High", 4000, 10, episodes=3)
    for name in ALGORITHM_NAMES:
        sim = run_with(configs, "High", 4000, 10, make_algorithm(name, q_table=q_table))
        s = sim.summary()
        assert len(s["names"]) == 25 and len(s["handled"]) == 25
        assert s["processed"] + s["dropped"] == s["total"], name


def test_ai_uses_all_servers_under_heavy_load():
    # Light traffic: the AI may favour the fastest servers (that is what it learned).
    # Heavy traffic: the fast servers fill up, so it must spread load over every server.
    configs = many_servers(25)
    q_table = train(configs, "High", 12000, 10, episodes=5)
    s = run_with(configs, "High", 12000, 10, make_algorithm("Q-Learning (AI)", q_table=q_table)).summary()
    used = sum(1 for a in s["assigned"] if a > 0)
    assert used == 25, "heavy load should reach all 25 servers, reached %d" % used


def test_weighted_round_robin_schedule_is_correct_after_failure():
    # the schedule is cached; it must be rebuilt when a server fails or recovers
    configs = default_configs()
    sim = Simulator(configs, [], WeightedRoundRobin(), duration=1)
    wrr = sim.algorithm
    first = [wrr.select(sim.servers).index for _ in range(10)]
    assert set(first) == {0, 1, 2, 3}
    sim.fail_server(1)
    after = [wrr.select(sim.servers).index for _ in range(40)]
    assert 1 not in after
    sim.recover_server(1)
    again = [wrr.select(sim.servers).index for _ in range(40)]
    assert 1 in again


def test_server_that_starts_failed_has_no_initial_load():
    configs = default_configs()
    configs[0].up = False
    configs[0].initial_load = 50
    sim = Simulator(configs, [], RoundRobin(), duration=1)
    assert len(sim.servers[0].active) == 0


def test_failure_mid_run_conserves_requests():
    configs = default_configs()
    sim = Simulator(copy.deepcopy(configs), generate_requests("High", 400, 4, 42), LeastConnections(), 4)
    for step in range(150):
        sim.step()
        if step == 50:
            sim.fail_server(2)
        if step == 100:
            sim.recover_server(2)
    sim.run_to_end()
    s = sim.summary()
    assert s["processed"] + s["dropped"] == s["total"]


if __name__ == "__main__":
    tests = [
        test_all_algorithms_normal_traffic,
        test_high_traffic,
        test_burst_traffic,
        test_server_overload_drops_requests,
        test_server_failure_and_recovery,
        test_failed_server_receives_no_new_requests,
        test_comparison_uses_same_scenario,
        test_ai_is_in_the_algorithm_list,
        test_ai_training_builds_a_q_table,
        test_ai_learning_changes_the_table_but_evaluation_does_not,
        test_ai_is_repeatable,
        test_ai_never_picks_a_failed_server,
        test_ai_with_all_servers_failed_drops_everything,
        test_ai_handles_untrained_table,
        test_server_state_values_are_in_range,
        test_25_servers_all_algorithms,
        test_ai_uses_all_servers_under_heavy_load,
        test_weighted_round_robin_schedule_is_correct_after_failure,
        test_server_that_starts_failed_has_no_initial_load,
        test_failure_mid_run_conserves_requests,
    ]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  PASSED")
    print("\nAll tests passed.")

# Network Load Balancing and Traffic Optimization System

**Team:**
1. Parth Mudpe (24BDS0353)
2. Rishit Bhagat (24BDS0152)

A desktop simulation of a network load balancer, built for a Computer Networks
college project. It shows how incoming client requests can be distributed
across multiple servers using five classic load-balancing strategies plus a
simple **AI strategy based on Q-Learning**, and lets you compare how each
strategy performs on the same traffic.

## Problem Statement

Modern web services run behind multiple servers. A load balancer decides
which server should handle each incoming request. The choice of algorithm
affects response time, server utilization, and how many requests get dropped
under heavy traffic. This project simulates that decision process so the
trade-offs between algorithms can be seen and measured directly, without
needing real servers or network hardware.

## Objectives

- Simulate client traffic under Normal, High and Burst conditions.
- Implement and compare five load-balancing algorithms.
- Add a simple AI method (tabular Q-Learning, no deep learning) that learns
  by trial and error which servers give short response times, and compare it
  with the five classic algorithms using the same metrics.
- Support from 3 up to 25 servers.
- Visualize server load, queue length, response time and throughput live.
- Let a server "fail" and "recover" during a simulation to show fault
  tolerance.
- Provide a fair, side-by-side comparison of all six methods on the
  identical traffic scenario.

## Technologies

- Python 3
- Tkinter (GUI)
- NumPy (traffic generation, numeric work)
- Matplotlib (charts, embedded in the Tkinter window)

No external services, sockets, or real network packets are used — this is a
discrete-time simulation.

## Methodology

The simulator advances in small fixed time steps (0.01 s). At each step:

1. Any requests whose arrival time has been reached are handed to the
   load-balancing algorithm, which picks a server (or drops the request if
   every server's queue is full).
2. Utilization, queue length and other measurements are sampled for the
   charts.
3. Every server processes its in-flight requests for that time slice.

Each server is modeled as a small pool of "slots" (parallel workers) plus a
FIFO waiting queue. `slots ≈ capacity / 10`, so a higher-capacity server can
process more requests at once. If a server's slots and queue are both full,
new requests to it are dropped.

## Algorithms

There are five classic algorithms and one AI algorithm.

1. **Round Robin** — cycles through available servers in order, skipping any
   that are currently failed.
2. **Weighted Round Robin** — same idea, but servers with higher capacity are
   picked proportionally more often.
3. **Least Connections** — sends each request to the available server with
   the fewest active + queued requests.
4. **Random Selection** — picks uniformly at random among available servers.
5. **Multi-Factor Dynamic Server Selection (proposed)** — scores every
   available server on five normalized factors and picks the best score:

   ```
   Score = 0.25 × Load Factor
         + 0.20 × Queue Factor
         + 0.20 × Response Time Factor
         + 0.20 × Capacity Factor
         + 0.15 × Availability Factor
   ```

   - **Load Factor** rewards servers with lower current load.
   - **Queue Factor** rewards servers with shorter queues.
   - **Response Time Factor** rewards servers with lower historical response
     time.
   - **Capacity Factor** rewards servers with more processing capacity.
   - **Availability Factor** rewards servers with a higher reliability
     rating.

   All five factors are normalized to a 0–1 range before weighting, so the
   formula stays simple to compute and simple to explain in a viva. The
   weights can be edited in `algorithms/multi_factor.py`.
6. **Q-Learning (AI)** — a simple reinforcement-learning method (see below).

## Q-Learning (AI)

File: `algorithms/q_learning.py` (about 180 lines including comments, only NumPy needed).

**Idea.** For every request, the balancer describes each available server with
four small numbers (its *state*), looks the state up in a small table (the
*Q-table*), and sends the request to the server whose state has the best
Q-value.

| Part | What it is here |
|---|---|
| State (of one server) | load level 0-4 (slots busy), queue level 0-4 (queue fullness), speed level 0-2 (capacity vs. the biggest server), latency level 0-1 (fast or slow base response time) |
| Action | "send this request to that server" |
| Reward | `- (response time in seconds)` when the request finishes, `-10` when it is dropped |
| Update | `Q(s) = Q(s) + 0.1 * (reward + 0.5 * Q(next s) - Q(s))` (next state = the same server just after it took the request) |
| Exploration | epsilon-greedy: 30% random choices at first, falling to 5% during training |

Because the state describes *one server* and not the whole system, the table
has only 150 possible entries and works unchanged for 3 to 25 servers.

**Training.** The AI practises on 25 short simulated runs of the chosen
traffic type and servers (different random traffic from the final test, so it
does not just memorise the test). This takes about 1-2 seconds. After that it
stops learning and only uses its table, so every comparison is fair and
repeatable. The GUI trains the AI automatically the first time you use it, and
again whenever the servers, traffic type, number of requests or duration
change.

This is a simplified form of Q-Learning chosen to stay easy to explain: the
state describes the candidate server, and the "next state" is that server's
state right after it accepted the request.

## Performance Metrics

Calculated directly from the simulation (no fabricated numbers):

(The AI is measured with exactly the same metrics as the other algorithms.)

- Average Response Time
- Throughput (requests processed per second)
- Average Server Utilization
- Average Queue Length
- Dropped Requests
- Load Distribution across servers

## How to Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with Tkinter (bundled with most Python installers on
Windows).

## How to Run

```bash
python main.py
```

## How to Run the Tests

```bash
python -m tests.test_simulation          # simulation, algorithms and Q-Learning (20 tests)
python -m tests.test_gui                 # clicks the real GUI functions (17 tests, needs a screen)
xvfb-run -a python -m tests.test_gui     # same, on a computer without a screen (Linux)
```

## How to Use the Application

1. **Simulation Settings** — choose a traffic type (Normal / High / Burst),
   number of requests, duration and algorithm, then click **Start
   Simulation**.
2. **Server Status** — watch each server's load, queue and connection count
   update live. Use **Fail Server** / **Recover Server** to demonstrate fault
   tolerance mid-simulation.
3. **Traffic Flow** — a simple diagram shows requests moving from the load
   balancer to whichever server just received one.
4. **Live Metrics** — headline numbers for the current run.
5. **Visualization tab** — switch between charts (requests per server,
   utilization, response time, queue length, load distribution).
6. **Comparison tab** — trains the AI, then runs the same traffic scenario
   through all six methods and shows a results table (the AI row is green), a
   short sentence saying where the AI ranks, and comparison charts.
7. **Configuration tab** — edit each server. **Add Server** / **Remove Last
   Server** change the number of servers (minimum 3, maximum 25); the list
   scrolls when there are many servers. Server names must be different. The
   buttons are locked while a simulation runs.
8. **AI** — choose **Q-Learning (AI)** in the Algorithm box. It is trained
   automatically before the first run (watch the status bar). Its server cards
   show the Q-value the AI gave each server.
9. **Run Demo** — loads a ready-made scenario and runs it automatically, for
   quick demonstration during evaluation.

## Sample Experiment

Dashboard settings: **High** traffic, **2000** requests, **10** seconds, the 4
default servers, then **Comparison -> Run Comparison**. Measured with this
version of the code (traffic seed 42, so you get the same numbers):

| Algorithm | Avg response (ms) | Throughput (req/s) | Utilization (%) | Avg queue | Dropped |
|---|---|---|---|---|---|
| Round Robin | 234.5 | 352.6 | 85.4 | 6.12 | 117 |
| Weighted Round Robin | 170.9 | 389.9 | 91.2 | 0.40 | 0 |
| Least Connections | 183.1 | 387.6 | 92.2 | 1.66 | 0 |
| Random Selection | 235.9 | 364.8 | 87.8 | 6.43 | 85 |
| Multi-Factor Dynamic Selection | 167.5 | 389.9 | 91.0 | 0.05 | 0 |
| Q-Learning (AI) | 166.9 | 389.9 | 92.4 | 0.06 | 0 |

Round Robin and Random Selection ignore server capacity, so the weaker servers
fill their queues and some requests are dropped. The three methods that take
server condition into account (Weighted Round Robin, Multi-Factor, Q-Learning)
drop nothing and keep queues short. Here the AI is about as good as
Multi-Factor, without hand-written weights; the 0.6 ms difference between them
is too small to call a win.

The result depends on the scenario. With very light traffic (for example 80
requests in 10 seconds) nothing queues and all methods are within a few
milliseconds; under extreme overload (for example High, 4000 requests) every
method drops a lot of requests, and Least Connections shows a lower *average*
response time partly because it drops more requests. Always read the dropped
column together with the response time.

## Limitations

- Time is simulated in discrete steps, not driven by real network I/O.
- Server behavior (processing speed, response time) is modeled with simple
  formulas, not measured from real hardware.
- The Multi-Factor formula uses fixed weights rather than weights learned
  from data.
- The AI is a small tabular Q-Learning agent. It learns from simulated
  traffic, so it is only as realistic as the simulation. It is trained on
  shortened runs without server failures, and it does not beat every classic
  algorithm in every scenario (see the sample experiment).
- Servers cannot be given custom speeds beyond the settings in the
  Configuration tab, and at least 3 servers are required.

## Future Scope

- Add persistent logging of simulation runs for later analysis.
- Support saving/loading server configurations and trained AI tables.
- Train the AI with random server failures, and try other reward settings.
- Extend the Multi-Factor method with adaptive weights based on observed
  performance.

## Project Structure

```
main.py                         start the application
algorithms/                     the six load-balancing methods (q_learning.py = the AI)
simulation/                     servers, requests/traffic, simulator, metrics
gui/dashboard.py                main window (tabs, server cards, configuration)
gui/charts.py                   all Matplotlib charts
tests/test_simulation.py        simulation + AI tests
tests/test_gui.py               GUI tests
```

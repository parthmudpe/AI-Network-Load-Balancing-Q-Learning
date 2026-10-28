"""
dashboard.py - The main window (Tkinter).

Tabs:
    Dashboard      -> settings, server cards, traffic flow, live metrics
    Charts         -> one Matplotlib chart at a time (choose from a dropdown)
    Comparison     -> run the same traffic through all 6 algorithms (5 classic + Q-Learning AI)
    Configuration  -> edit the servers (3 to 25) and the Multi-Factor weights
"""
import copy
import math
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms import (ALGORITHM_NAMES, AI_NAME, make_algorithm, DEFAULT_WEIGHTS,
                        train_q_learning)
from simulation.request import TRAFFIC_MODES, generate_requests
from simulation.server import ServerConfig, default_configs
from simulation.simulator import Simulator
from algorithms.q_learning import TRAIN_EPISODES
from gui.charts import CHART_NAMES, COMPARE_METRICS, draw_chart

# ---------- colors and fonts ----------
BG = "#F3F6FA"
CARD = "#FFFFFF"
BORDER = "#D6DDE8"
BLUE = "#2563EB"
DARK_BLUE = "#1E3A8A"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"
GRAY = "#6B7280"
TEXT = "#1F2937"
FONT = "Segoe UI"

MAX_SERVERS = 25
MIN_SERVERS = 3
TRAFFIC_SEED = 42        # same seed -> same traffic every time


class ScrollFrame(tk.Frame):
    """A frame with a vertical scrollbar. Put your widgets into  scroll_frame.inner ."""

    def __init__(self, parent, bg=BG, height=100):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, height=height)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        # keep the scroll area as big as the content, and the content as wide as the canvas
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._window, width=e.width))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Network Load Balancing and Traffic Optimization System")
        self.geometry("1280x820")
        self.minsize(1180, 760)
        self.configure(bg=BG)

        # ----- data -----
        self.configs = default_configs()          # server settings
        self.weights = dict(DEFAULT_WEIGHTS)      # Multi-Factor weights
        self.sim = None                           # current Simulator (None = idle)
        self.running = False
        self.steps_per_frame = 5                  # simulation speed
        self.frame_count = 0
        self.comparison = None                    # results of the last comparison
        self.card_columns = 2                     # server cards per row (3 when there are many)
        self.q_table = None                       # what the Q-Learning AI has learned
        self.q_signature = None                   # the scenario the AI was trained for
        self.bind_all("<MouseWheel>", self.on_mousewheel)      # Windows / macOS
        self.bind_all("<Button-4>", self.on_mousewheel)        # Linux
        self.bind_all("<Button-5>", self.on_mousewheel)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_styles()
        self.build_header()
        self.build_tabs()
        self.build_footer()
        self.refresh_dashboard()

    # =====================================================================
    # Styles
    # =====================================================================
    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=(FONT, 11, "bold"), padding=(18, 8),
                        background="#E5EAF2", foreground=GRAY)
        style.map("TNotebook.Tab", background=[("selected", CARD)],
                  foreground=[("selected", BLUE)])
        style.configure("TCombobox", padding=4)
        style.configure("Treeview", font=(FONT, 11), rowheight=30, background=CARD,
                        fieldbackground=CARD, foreground=TEXT)
        style.configure("Treeview.Heading", font=(FONT, 10, "bold"), background="#E5EAF2",
                        foreground=TEXT, padding=6)
        for name, color in (("Green", GREEN), ("Orange", ORANGE), ("Red", RED)):
            style.configure(name + ".Horizontal.TProgressbar", troughcolor="#E5EAF2",
                            background=color, thickness=14, borderwidth=0)

    def on_close(self):
        """Window closed: stop the animation first so no timer fires after the window is gone."""
        self.running = False
        self.update()
        self.destroy()

    def on_mousewheel(self, event):
        """Scroll the ScrollFrame that is under the mouse pointer (if there is one)."""
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):          # e.g. an open dropdown list
            return
        while widget is not None and not isinstance(widget, ScrollFrame):
            widget = widget.master
        if widget is None:
            return
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        widget.canvas.yview_scroll(step * 2, "units")

    def make_card(self, parent, **grid_or_pack):
        return tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)

    def make_button(self, parent, text, command, color=BLUE, width=None):
        return tk.Button(parent, text=text, command=command, bg=color, fg="white",
                         activebackground=color, activeforeground="white",
                         font=(FONT, 10, "bold"), relief="flat", cursor="hand2",
                         padx=14, pady=6, width=width, bd=0, disabledforeground="#E5E7EB")

    # =====================================================================
    # Header / footer
    # =====================================================================
    def build_header(self):
        header = tk.Frame(self, bg=DARK_BLUE)
        header.pack(fill="x")
        tk.Label(header, text="Network Load Balancing and Traffic Optimization System",
                 font=(FONT, 19, "bold"), bg=DARK_BLUE, fg="white").pack(anchor="w", padx=20, pady=(10, 0))
        tk.Label(header, text="Traffic Simulation and Performance Analysis",
                 font=(FONT, 11), bg=DARK_BLUE, fg="#BFDBFE").pack(anchor="w", padx=20, pady=(0, 10))

    def build_footer(self):
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready. Choose your settings and click Start Simulation.")
        tk.Label(footer, textvariable=self.status_var, font=(FONT, 10), bg=BG, fg=GRAY
                 ).pack(side="left", padx=20, pady=6)
        tk.Label(footer, text="Team: Parth Mudpe (24BDS0353)  |  Rishit Bhagat (24BDS0152)",
                 font=(FONT, 9), bg=BG, fg=GRAY).pack(side="right", padx=20)

    def build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(10, 0))
        self.tab_dashboard = tk.Frame(self.tabs, bg=BG)
        self.tab_charts = tk.Frame(self.tabs, bg=BG)
        self.tab_compare = tk.Frame(self.tabs, bg=BG)
        self.tab_config = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(self.tab_dashboard, text="Dashboard")
        self.tabs.add(self.tab_charts, text="Charts")
        self.tabs.add(self.tab_compare, text="Comparison")
        self.tabs.add(self.tab_config, text="Configuration")

        self.build_dashboard_tab()
        self.build_charts_tab()
        self.build_comparison_tab()
        self.build_config_tab()
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.redraw_chart())

    # =====================================================================
    # DASHBOARD TAB
    # =====================================================================
    def build_dashboard_tab(self):
        tab = self.tab_dashboard
        self.build_settings_section(tab)

        middle = tk.Frame(tab, bg=BG)
        middle.pack(fill="both", expand=True, pady=(10, 0))
        middle.columnconfigure(0, weight=3)
        middle.columnconfigure(1, weight=2)
        middle.rowconfigure(0, weight=1)

        # Section 2: server status (cards are created in build_server_cards)
        left = tk.Frame(middle, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(left, text="2. Server Status", font=(FONT, 12, "bold"), bg=BG, fg=TEXT
                 ).pack(anchor="w", pady=(0, 4))
        self.cards_scroll = ScrollFrame(left, bg=BG)           # scrolls when there are many servers
        self.cards_scroll.pack(fill="both", expand=True)
        self.cards_area = self.cards_scroll.inner

        # Section 3: traffic flow
        right = tk.Frame(middle, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(right, text="3. Traffic Flow", font=(FONT, 12, "bold"), bg=BG, fg=TEXT
                 ).pack(anchor="w", pady=(0, 4))
        flow_card = self.make_card(right)
        flow_card.pack(fill="both", expand=True)
        self.flow_canvas = tk.Canvas(flow_card, bg=CARD, highlightthickness=0)
        self.flow_canvas.pack(fill="both", expand=True)
        self.flow_canvas.bind("<Configure>", lambda e: self.draw_flow())

        # Section 4: live metrics
        tk.Label(tab, text="4. Live Metrics", font=(FONT, 12, "bold"), bg=BG, fg=TEXT
                 ).pack(anchor="w", pady=(8, 4))
        metrics_row = tk.Frame(tab, bg=BG)
        metrics_row.pack(fill="x", pady=(0, 8))
        self.metric_labels = {}
        titles = [("total", "Total Requests"), ("processed", "Processed"),
                  ("dropped", "Dropped"), ("response", "Avg Response Time"),
                  ("throughput", "Throughput"), ("utilization", "Avg Utilization")]
        for col, (key, title) in enumerate(titles):
            metrics_row.columnconfigure(col, weight=1, uniform="metric")
            card = self.make_card(metrics_row)
            card.grid(row=0, column=col, sticky="nsew", padx=4)
            tk.Label(card, text=title, font=(FONT, 10), bg=CARD, fg=GRAY).pack(pady=(8, 0))
            value = tk.Label(card, text="0", font=(FONT, 18, "bold"), bg=CARD, fg=DARK_BLUE)
            value.pack(pady=(0, 8))
            self.metric_labels[key] = value

    def build_settings_section(self, parent):
        card = self.make_card(parent)
        card.pack(fill="x")
        tk.Label(card, text="1. Simulation Settings", font=(FONT, 12, "bold"), bg=CARD, fg=TEXT
                 ).grid(row=0, column=0, columnspan=8, sticky="w", padx=14, pady=(8, 2))

        self.traffic_var = tk.StringVar(value="Normal")
        self.requests_var = tk.StringVar(value="2000")
        self.duration_var = tk.StringVar(value="10")
        self.algorithm_var = tk.StringVar(value=ALGORITHM_NAMES[0])

        def label(text, col):
            tk.Label(card, text=text, font=(FONT, 10), bg=CARD, fg=GRAY
                     ).grid(row=1, column=col, sticky="w", padx=(14, 4))

        label("Traffic Type", 0)
        label("Number of Requests", 1)
        label("Duration (seconds)", 2)
        label("Algorithm", 3)

        self.traffic_box = ttk.Combobox(card, textvariable=self.traffic_var, values=TRAFFIC_MODES,
                                        state="readonly", width=10, font=(FONT, 11))
        self.requests_box = ttk.Spinbox(card, textvariable=self.requests_var, from_=10, to=20000,
                                        increment=100, width=9, font=(FONT, 11))
        self.duration_box = ttk.Spinbox(card, textvariable=self.duration_var, from_=1, to=120,
                                        increment=1, width=8, font=(FONT, 11))
        self.algorithm_box = ttk.Combobox(card, textvariable=self.algorithm_var,
                                          values=ALGORITHM_NAMES, state="readonly", width=30,
                                          font=(FONT, 11))
        self.algorithm_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_dashboard())
        self.traffic_box.grid(row=2, column=0, padx=(14, 4), pady=(0, 10), sticky="w")
        self.requests_box.grid(row=2, column=1, padx=(14, 4), pady=(0, 10), sticky="w")
        self.duration_box.grid(row=2, column=2, padx=(14, 4), pady=(0, 10), sticky="w")
        self.algorithm_box.grid(row=2, column=3, padx=(14, 4), pady=(0, 10), sticky="w")

        button_area = tk.Frame(card, bg=CARD)
        button_area.grid(row=1, column=4, rowspan=2, padx=(30, 14), pady=(0, 10), sticky="e")
        card.columnconfigure(4, weight=1)
        self.start_button = self.make_button(button_area, "Start Simulation", self.start_simulation, GREEN)
        self.reset_button = self.make_button(button_area, "Reset", self.reset_simulation, GRAY)
        self.demo_button = self.make_button(button_area, "Run Demo", self.run_demo, BLUE)
        self.start_button.pack(side="left", padx=4)
        self.reset_button.pack(side="left", padx=4)
        self.demo_button.pack(side="left", padx=4)

        self.setting_widgets = [self.traffic_box, self.requests_box, self.duration_box,
                                self.algorithm_box]

    # ---------- server cards ----------
    def build_server_cards(self):
        """Create one card per server (called again when the server count changes)."""
        for child in self.cards_area.winfo_children():
            child.destroy()
        self.cards = []
        columns = 2 if len(self.configs) <= 6 else 3          # more servers -> more columns
        self.card_columns = columns                            # (3 columns -> shorter texts)
        for c in range(3):
            self.cards_area.columnconfigure(c, weight=1 if c < columns else 0,
                                            uniform="cards" if c < columns else "")

        for i in range(len(self.configs)):
            frame = self.make_card(self.cards_area)
            frame.grid(row=i // columns, column=i % columns, sticky="nsew", padx=4, pady=4)

            top = tk.Frame(frame, bg=CARD)
            top.pack(fill="x", padx=12, pady=(8, 2))
            name = tk.Label(top, text="", font=(FONT, 12, "bold"), bg=CARD, fg=TEXT)
            name.pack(side="left")
            badge = tk.Label(top, text="AVAILABLE", font=(FONT, 9, "bold"), fg="white",
                             bg=GREEN, padx=8, pady=1)
            badge.pack(side="right")

            bar_row = tk.Frame(frame, bg=CARD)
            bar_row.pack(fill="x", padx=12, pady=2)
            tk.Label(bar_row, text="Load", font=(FONT, 10), bg=CARD, fg=GRAY).pack(side="left")
            bar = ttk.Progressbar(bar_row, maximum=100, style="Green.Horizontal.TProgressbar")
            bar.pack(side="left", fill="x", expand=True, padx=8)
            load_label = tk.Label(bar_row, text="0%", font=(FONT, 10, "bold"), bg=CARD, fg=TEXT, width=5)
            load_label.pack(side="left")

            info1 = tk.Label(frame, text="", font=(FONT, 10), bg=CARD, fg=TEXT, anchor="w")
            info1.pack(fill="x", padx=12)
            info2 = tk.Label(frame, text="", font=(FONT, 10), bg=CARD, fg=TEXT, anchor="w")
            info2.pack(fill="x", padx=12)

            bottom = tk.Frame(frame, bg=CARD)
            bottom.pack(fill="x", padx=12, pady=(2, 8))
            score = tk.Label(bottom, text="", font=(FONT, 10, "bold"), bg=CARD, fg=BLUE)
            score.pack(side="left")
            button = tk.Button(bottom, text="Fail Server", font=(FONT, 9, "bold"), bg="#FEE2E2",
                               fg=RED, relief="flat", cursor="hand2", padx=10, pady=2,
                               activebackground="#FECACA",
                               command=lambda idx=i: self.toggle_server(idx))
            button.pack(side="right")

            self.cards.append({"name": name, "badge": badge, "bar": bar, "load": load_label,
                               "info1": info1, "info2": info2, "score": score, "button": button})

    # =====================================================================
    # Updating the dashboard
    # =====================================================================
    def server_views(self):
        """One dictionary of display values per server (from the simulator, or from the settings)."""
        views = []
        if self.sim is not None:
            summary = self.sim.metrics.summary(self.sim.time)
            for i, s in enumerate(self.sim.servers):
                measured = summary["server_response"][i]
                views.append({"name": s.name, "up": s.available, "capacity": s.capacity,
                              "load": s.load_percent, "queue": len(s.queue),
                              "max_queue": s.max_queue, "connections": s.connections,
                              "response": measured if measured > 0 else s.response_time,
                              "score": s.last_score,
                              "assigned": int(summary["assigned"][i])})
        else:
            for c in self.configs:
                slots = max(1, round(c.capacity / 10))
                busy = round(c.initial_load / 100 * slots)
                views.append({"name": c.name, "up": c.up, "capacity": c.capacity,
                              "load": 100.0 * busy / slots, "queue": 0, "max_queue": c.max_queue,
                              "connections": busy, "response": c.response_time,
                              "score": None, "assigned": 0})
        return views

    def current_algorithm_name(self):
        if self.sim is not None:
            return self.sim.algorithm.name
        return self.algorithm_var.get()

    def refresh_dashboard(self):
        """Redraw cards, flow diagram and metrics."""
        if len(self.configs) != len(getattr(self, "cards", [])):
            self.build_server_cards()
        views = self.server_views()
        algorithm_name = self.current_algorithm_name()
        show_score = algorithm_name in ("Multi-Factor Dynamic Selection", AI_NAME)
        score_label = "Q-value" if algorithm_name == AI_NAME else "Score"

        compact = self.card_columns == 3          # narrow cards need shorter texts
        for card, v in zip(self.cards, views):
            card["name"].config(text=v["name"])
            if v["up"]:
                card["badge"].config(text="UP" if compact else "AVAILABLE", bg=GREEN)
                card["button"].config(text="Fail" if compact else "Fail Server", bg="#FEE2E2", fg=RED)
            else:
                card["badge"].config(text="FAILED", bg=RED)
                card["button"].config(text="Recover" if compact else "Recover Server",
                                      bg="#DCFCE7", fg=GREEN)
            card["bar"].config(value=v["load"])
            level = "Green" if v["load"] < 60 else ("Orange" if v["load"] < 85 else "Red")
            card["bar"].config(style=level + ".Horizontal.TProgressbar")
            card["load"].config(text="%d%%" % round(v["load"]))
            if compact:
                card["info1"].config(text="Cap: %d/s  Resp: %d ms" % (v["capacity"], v["response"]))
                card["info2"].config(text="Queue: %d/%d  Conn: %d"
                                     % (v["queue"], v["max_queue"], v["connections"]))
            else:
                card["info1"].config(text="Capacity: %d req/s   Resp: %d ms"
                                     % (v["capacity"], v["response"]))
                card["info2"].config(text="Queue: %d / %d   Conn: %d"
                                     % (v["queue"], v["max_queue"], v["connections"]))
            if show_score:
                label = ("Q" if score_label == "Q-value" else score_label) if compact else score_label
                if v["score"] is None:
                    text = "%s: -" % label
                else:
                    text = ("%s: %.3f" if label == "Score" else "%s: %.2f") % (label, v["score"])
                card["score"].config(text=text)
            else:
                card["score"].config(text="")
            # Fail/Recover only works when idle (edits the start state) or while running
            can_click = self.sim is None or not self.sim.finished
            card["button"].config(state="normal" if can_click else "disabled")

        self.draw_flow(views)
        self.update_metrics()

    def update_metrics(self):
        if self.sim is None:
            values = {"total": "0", "processed": "0", "dropped": "0", "response": "0 ms",
                      "throughput": "0 req/s", "utilization": "0%"}
        else:
            s = self.sim.summary()
            values = {"total": str(s["total"]), "processed": str(s["processed"]),
                      "dropped": str(s["dropped"]), "response": "%.0f ms" % s["avg_response"],
                      "throughput": "%.0f req/s" % s["throughput"],
                      "utilization": "%.0f%%" % s["avg_utilization"]}
        for key, text in values.items():
            self.metric_labels[key].config(text=text)
        self.metric_labels["dropped"].config(fg=RED if values["dropped"] != "0" else DARK_BLUE)

    def draw_flow(self, views=None):
        """Simple diagram: CLIENT REQUESTS -> LOAD BALANCER -> servers."""
        c = self.flow_canvas
        c.delete("all")
        if views is None:
            views = self.server_views()
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 300)
        n = len(views)
        compact = n > 10                      # many servers: smaller boxes, short labels
        cx = w / 2
        total_sent = sum(v["assigned"] for v in views)
        recent = self.sim.recent if self.sim is not None else set()

        def box(x1, y1, x2, y2, text, fill, fg, outline, size=10):
            c.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2 if not compact else 1)
            c.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill=fg,
                          font=(FONT, size, "bold"), justify="center")

        client_y, lb_y = 18, h * 0.36
        server_y = h - (60 if compact else 82)
        box(cx - 90, client_y, cx + 90, client_y + 38, "CLIENT REQUESTS", "#E0E7FF", DARK_BLUE, BLUE)
        c.create_line(cx, client_y + 38, cx, lb_y, arrow="last", width=3, fill=BLUE, arrowshape=(12, 14, 5))
        box(cx - 90, lb_y, cx + 90, lb_y + 38, "LOAD BALANCER", BLUE, "white", DARK_BLUE)

        slot = (w - 20) / n
        box_w = max(8, slot - 3) if compact else max(8, min(96, slot - 10))
        box_h = 28 if compact else 52
        for i, v in enumerate(views):
            x = 10 + slot * (i + 0.5)
            share = v["assigned"] / total_sent if total_sent else 0
            if not v["up"]:
                line_color, dash = "#9CA3AF", (4, 4)
            else:
                line_color, dash = (ORANGE if i in recent else "#60A5FA"), None
            line_width = (1 + 14 * share) if compact else (2 + 7 * share)
            c.create_line(cx, lb_y + 38, x, server_y, arrow="last", fill=line_color,
                          width=line_width, dash=dash,
                          arrowshape=(6, 8, 3) if compact else (10, 12, 4))
            fill = "#DCFCE7" if v["up"] else "#FEE2E2"
            outline = GREEN if v["up"] else RED
            short = "S%d" % (i + 1)
            if compact:
                box(x - box_w / 2, server_y, x + box_w / 2, server_y + box_h, short,
                    fill, TEXT, outline, size=7)
            else:
                state = "%d req" % v["assigned"] if v["up"] else "FAILED"
                box(x - box_w / 2, server_y, x + box_w / 2, server_y + box_h,
                    short + "\n" + state, fill, TEXT, outline)
                c.create_text(x, server_y + 66, text="%d%% of load" % round(share * 100),
                              fill=GRAY, font=(FONT, 9))
        if compact:
            c.create_text(cx, h - 14, text="%d servers (red = failed). Details are on the server cards." % n,
                          fill=GRAY, font=(FONT, 9))

    # =====================================================================
    # Simulation control
    # =====================================================================
    def read_settings(self):
        """Read and check the settings. Returns (traffic, requests, duration) or None."""
        try:
            num_requests = int(float(self.requests_var.get()))
            duration = float(self.duration_var.get())
        except (ValueError, OverflowError):
            messagebox.showerror("Invalid input", "Number of Requests and Duration must be numbers.")
            return None
        if num_requests < 1 or num_requests > 20000:
            messagebox.showerror("Invalid input", "Number of Requests must be between 1 and 20000.")
            return None
        if duration < 1 or duration > 120:
            messagebox.showerror("Invalid input", "Duration must be between 1 and 120 seconds.")
            return None
        return self.traffic_var.get(), num_requests, duration

    def ensure_ai_trained(self, traffic, num_requests, duration):
        """
        Train the Q-Learning AI if it has not been trained for THIS scenario yet
        (same servers, traffic type, number of requests and duration).
        Training is a few quick practice simulations, so it takes only a second or two.
        """
        servers = tuple((c.name, c.capacity, c.max_queue, c.initial_load, c.response_time,
                         c.availability, c.up) for c in self.configs)
        signature = (traffic, num_requests, duration, servers)
        if self.q_table is not None and signature == self.q_signature:
            return                                    # already trained for this scenario

        def progress(done, total):
            self.status_var.set("Training the AI (Q-Learning): practice run %d of %d ..." % (done, total))
            self.update_idletasks()

        self.config(cursor="watch")
        try:
            self.q_table = train_q_learning(self.configs, traffic, num_requests, duration,
                                            progress=progress)
        finally:
            self.config(cursor="")
        self.q_signature = signature
        self.status_var.set("AI trained: %d practice runs, %d server states learned."
                            % (TRAIN_EPISODES, len(self.q_table)))

    def start_simulation(self, slow=False):
        if self.running:
            return
        settings = self.read_settings()
        if settings is None:
            return
        traffic, num_requests, duration = settings
        if not any(c.up for c in self.configs):
            messagebox.showwarning("No server", "All servers are FAILED. Recover at least one server first.")
            return

        if self.algorithm_var.get() == AI_NAME:
            self.ensure_ai_trained(traffic, num_requests, duration)
        requests = generate_requests(traffic, num_requests, duration, TRAFFIC_SEED)
        algorithm = make_algorithm(self.algorithm_var.get(), self.weights, q_table=self.q_table)
        self.sim = Simulator(copy.deepcopy(self.configs), requests, algorithm, duration)
        self.running = True
        self.steps_per_frame = 1 if slow else 5      # demo runs in slow motion, normal runs do not
        self.frame_count = 0
        self.set_controls_enabled(False)
        self.status_var.set("Running: %s traffic, %d requests, %s"
                            % (traffic, num_requests, algorithm.name))
        self.after(30, self.run_frame)

    def run_frame(self):
        """Called about 30 times per second while the simulation runs."""
        if not self.running or self.sim is None:
            return
        for _ in range(self.steps_per_frame):
            self.sim.step()
            if self.sim.finished:
                break
        self.frame_count += 1
        self.refresh_dashboard()
        if self.tabs.select() == str(self.tab_charts) and self.frame_count % 15 == 0:
            self.redraw_chart()

        if self.sim.finished:
            self.running = False
            self.set_controls_enabled(True)
            self.status_var.set("Finished at t = %.1f s. Open the Charts tab to see the results."
                                % self.sim.time)
            self.refresh_dashboard()
            self.redraw_chart()
        else:
            self.status_var.set("Running... simulated time = %.1f s   (%d / %d requests arrived)"
                                % (self.sim.time, self.sim.next_request, len(self.sim.requests)))
            self.after(30, self.run_frame)

    def reset_simulation(self):
        self.running = False
        self.sim = None
        self.steps_per_frame = 5
        self.set_controls_enabled(True)
        self.status_var.set("Reset. Ready for a new simulation.")
        self.refresh_dashboard()
        self.redraw_chart()

    def run_demo(self):
        """Load a ready-made scenario and run it slowly so the distribution is visible."""
        if self.running:
            return
        self.reset_simulation()
        self.configs = default_configs()                # 4 servers, mixed capacities
        self.traffic_var.set("Burst")
        self.requests_var.set("100")
        self.duration_var.set("3")
        self.refresh_dashboard()
        self.build_config_rows()
        self.start_simulation(slow=True)
        if self.running:
            self.status_var.set("DEMO: 4 servers, 100 requests, Burst traffic, %s. "
                                "Try clicking 'Fail Server' while it runs!" % self.algorithm_var.get())

    def toggle_server(self, index):
        """Fail / Recover button on a server card."""
        if self.sim is not None and not self.sim.finished:
            server = self.sim.servers[index]
            if server.available:
                self.sim.fail_server(index)
            else:
                self.sim.recover_server(index)
        elif self.sim is None:
            self.configs[index].up = not self.configs[index].up
            if index < len(self.config_vars):            # keep the Configuration tab in step
                self.config_vars[index]["up"].set(self.configs[index].up)
        self.refresh_dashboard()

    def set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.start_button.config(state=state)
        self.demo_button.config(state=state)
        for widget in self.setting_widgets:
            widget.config(state="readonly" if (enabled and isinstance(widget, ttk.Combobox)) else state)
        self.apply_button.config(state=state)
        self.compare_button.config(state=state)
        for button in self.config_buttons:               # Add / Remove / Restore Defaults
            button.config(state=state)

    # =====================================================================
    # CHARTS TAB
    # =====================================================================
    def build_charts_tab(self):
        tab = self.tab_charts
        top = tk.Frame(tab, bg=BG)
        top.pack(fill="x", pady=(10, 6))
        tk.Label(top, text="5. Visualization", font=(FONT, 12, "bold"), bg=BG, fg=TEXT
                 ).pack(side="left", padx=(4, 16))
        tk.Label(top, text="Chart:", font=(FONT, 10), bg=BG, fg=GRAY).pack(side="left")
        self.chart_var = tk.StringVar(value=CHART_NAMES[0])
        chart_box = ttk.Combobox(top, textvariable=self.chart_var, values=CHART_NAMES,
                                 state="readonly", width=34, font=(FONT, 11))
        chart_box.pack(side="left", padx=8)
        chart_box.bind("<<ComboboxSelected>>", lambda e: self.redraw_chart())

        card = self.make_card(tab)
        card.pack(fill="both", expand=True, pady=(0, 8))
        self.chart_figure = Figure(figsize=(9, 5), dpi=100, facecolor="white")
        self.chart_canvas = FigureCanvasTkAgg(self.chart_figure, master=card)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def redraw_chart(self):
        summary = self.sim.summary() if self.sim is not None else None
        draw_chart(self.chart_figure, self.chart_var.get(), summary,
                   self.comparison, self.compare_metric_var.get())
        self.chart_canvas.draw_idle()
        self.redraw_compare_chart()

    # =====================================================================
    # COMPARISON TAB
    # =====================================================================
    def build_comparison_tab(self):
        tab = self.tab_compare
        top = self.make_card(tab)
        top.pack(fill="x", pady=(10, 8))
        tk.Label(top, text="Algorithm Comparison", font=(FONT, 12, "bold"), bg=CARD, fg=TEXT
                 ).pack(side="left", padx=14, pady=10)
        # the button is packed BEFORE the long description, so it can never be pushed out of view
        self.compare_button = self.make_button(top, "Run Comparison", self.run_comparison, BLUE)
        self.compare_button.pack(side="right", padx=14, pady=8)
        tk.Label(top, text="Same servers and same traffic for all 6 algorithms (the AI is trained first).",
                 font=(FONT, 10), bg=CARD, fg=GRAY).pack(side="left", padx=6)

        columns = ("algorithm", "response", "throughput", "utilization", "queue", "dropped")
        headings = ("Algorithm", "Avg Response (ms)", "Throughput (req/s)", "Utilization (%)",
                    "Avg Queue", "Dropped")
        self.compare_table = ttk.Treeview(tab, columns=columns, show="headings",
                                          height=len(ALGORITHM_NAMES))
        for col, heading in zip(columns, headings):
            self.compare_table.heading(col, text=heading)
            self.compare_table.column(col, anchor="center", width=140)
        self.compare_table.column("algorithm", anchor="w", width=260)
        self.compare_table.tag_configure("ai", background="#DCFCE7")     # AI row is green
        self.compare_table.pack(fill="x", pady=(0, 4))
        self.compare_note = tk.Label(tab, text="", font=(FONT, 10), bg=BG, fg=TEXT,
                                     anchor="w", justify="left")
        self.compare_note.pack(fill="x", padx=4, pady=(0, 6))

        row = tk.Frame(tab, bg=BG)
        row.pack(fill="x")
        tk.Label(row, text="Compare by:", font=(FONT, 10), bg=BG, fg=GRAY).pack(side="left", padx=4)
        self.compare_metric_var = tk.StringVar(value="Avg Response Time")
        metric_box = ttk.Combobox(row, textvariable=self.compare_metric_var,
                                  values=list(COMPARE_METRICS.keys()), state="readonly",
                                  width=22, font=(FONT, 11))
        metric_box.pack(side="left", padx=8)
        metric_box.bind("<<ComboboxSelected>>", lambda e: self.redraw_chart())

        card = self.make_card(tab)
        card.pack(fill="both", expand=True, pady=(6, 8))
        self.compare_figure = Figure(figsize=(9, 3.6), dpi=100, facecolor="white")
        self.compare_canvas = FigureCanvasTkAgg(self.compare_figure, master=card)
        self.compare_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def run_comparison(self):
        settings = self.read_settings()
        if settings is None:
            return
        traffic, num_requests, duration = settings
        if not any(c.up for c in self.configs):
            messagebox.showwarning("No server", "All servers are FAILED. Recover at least one server first.")
            return

        self.ensure_ai_trained(traffic, num_requests, duration)      # the AI practises first
        self.config(cursor="watch")
        self.status_var.set("Running comparison...")
        self.update_idletasks()
        results = []
        for name in ALGORITHM_NAMES:
            # SAME traffic (same seed), SAME servers (fresh copy of the settings)
            requests = generate_requests(traffic, num_requests, duration, TRAFFIC_SEED)
            sim = Simulator(copy.deepcopy(self.configs), requests,
                            make_algorithm(name, self.weights, q_table=self.q_table), duration)
            sim.run_to_end()
            results.append({"algorithm": name, "summary": sim.summary()})
        self.comparison = results
        self.config(cursor="")

        for row in self.compare_table.get_children():
            self.compare_table.delete(row)
        for r in results:
            s = r["summary"]
            self.compare_table.insert("", "end", tags=("ai",) if r["algorithm"] == AI_NAME else (),
                                      values=(
                r["algorithm"], "%.1f" % s["avg_response"], "%.1f" % s["throughput"],
                "%.1f" % s["avg_utilization"], "%.2f" % s["avg_queue"], s["dropped"]))
        self.compare_note.config(text=self.comparison_summary(results))
        self.status_var.set("Comparison finished: %s traffic, %d requests, %d s."
                            % (traffic, num_requests, duration))
        self.redraw_chart()

    @staticmethod
    def comparison_summary(results):
        """One sentence that says where the AI ranks (calculated from the table, nothing assumed)."""
        names = [r["algorithm"] for r in results]
        by_response = sorted(results, key=lambda r: r["summary"]["avg_response"])
        by_drops = sorted(results, key=lambda r: r["summary"]["dropped"])
        ai = names.index(AI_NAME)
        ai_summary = results[ai]["summary"]
        rank_response = [r["algorithm"] for r in by_response].index(AI_NAME) + 1
        rank_drops = [r["algorithm"] for r in by_drops].index(AI_NAME) + 1
        return ("Lowest average response time: %s (%.1f ms).   Fewest dropped requests: %s (%d).\n"
                "Q-Learning (AI): rank %d of %d for response time (%.1f ms), rank %d of %d for dropped "
                "requests (%d).\nSmall differences can be noise, and an algorithm that drops more requests "
                "can show a lower average response time."
                % (by_response[0]["algorithm"], by_response[0]["summary"]["avg_response"],
                   by_drops[0]["algorithm"], by_drops[0]["summary"]["dropped"],
                   rank_response, len(results), ai_summary["avg_response"],
                   rank_drops, len(results), ai_summary["dropped"]))

    def redraw_compare_chart(self):
        draw_chart(self.compare_figure, "Algorithm Comparison", None, self.comparison,
                   self.compare_metric_var.get())
        self.compare_canvas.draw_idle()

    # =====================================================================
    # CONFIGURATION TAB
    # =====================================================================
    def build_config_tab(self):
        tab = self.tab_config
        card = self.make_card(tab)
        card.pack(fill="both", expand=True, pady=(10, 8))
        tk.Label(card, text="Server Configuration", font=(FONT, 12, "bold"), bg=CARD, fg=TEXT
                 ).pack(anchor="w", padx=14, pady=(10, 0))
        tk.Label(card, text="Change the values, then click 'Apply Changes'. You can have %d to %d servers. "
                            "Availability %% is the server's reliability rating (used by Multi-Factor)."
                            % (MIN_SERVERS, MAX_SERVERS),
                 font=(FONT, 10), bg=CARD, fg=GRAY).pack(anchor="w", padx=14, pady=(0, 6))
        self.config_scroll = ScrollFrame(card, bg=CARD, height=190)    # scrolls with many servers
        self.config_scroll.pack(fill="both", expand=True, padx=14, pady=4)
        self.config_grid = self.config_scroll.inner

        buttons = tk.Frame(card, bg=CARD)
        buttons.pack(fill="x", padx=14, pady=(6, 12))
        self.apply_button = self.make_button(buttons, "Apply Changes", self.apply_config, GREEN)
        self.apply_button.pack(side="left", padx=(0, 8))
        self.config_buttons = [
            self.make_button(buttons, "Add Server", self.add_server, BLUE),
            self.make_button(buttons, "Remove Last Server", self.remove_server, GRAY),
            self.make_button(buttons, "Restore Defaults", self.restore_defaults, GRAY),
        ]
        for button in self.config_buttons:
            button.pack(side="left", padx=4)
        self.server_count_label = tk.Label(buttons, text="", font=(FONT, 10, "bold"), bg=CARD, fg=GRAY)
        self.server_count_label.pack(side="right", padx=8)

        # Multi-Factor weights
        wcard = self.make_card(tab)
        wcard.pack(fill="x")
        tk.Label(wcard, text="Multi-Factor Weights", font=(FONT, 12, "bold"), bg=CARD, fg=TEXT
                 ).grid(row=0, column=0, columnspan=6, sticky="w", padx=14, pady=(10, 0))
        tk.Label(wcard, text="Score = w1 x Load + w2 x Queue + w3 x Response + w4 x Capacity + "
                             "w5 x Availability      (weights are divided by their total, so any values work)",
                 font=(FONT, 10), bg=CARD, fg=GRAY).grid(row=1, column=0, columnspan=6, sticky="w",
                                                         padx=14, pady=(0, 6))
        self.weight_vars = {}
        names = [("load", "Load"), ("queue", "Queue"), ("response", "Response Time"),
                 ("capacity", "Capacity"), ("availability", "Availability")]
        for col, (key, title) in enumerate(names):
            tk.Label(wcard, text=title, font=(FONT, 10), bg=CARD, fg=GRAY
                     ).grid(row=2, column=col, sticky="w", padx=(14, 4))
            var = tk.StringVar(value=str(self.weights[key]))
            tk.Entry(wcard, textvariable=var, width=8, font=(FONT, 11), justify="center"
                     ).grid(row=3, column=col, sticky="w", padx=(14, 4), pady=(0, 12))
            self.weight_vars[key] = var

        self.config_vars = []
        self.build_config_rows()

    def build_config_rows(self):
        """Draw one row of entry boxes for every server."""
        for child in self.config_grid.winfo_children():
            child.destroy()
        self.config_vars = []
        self.server_count_label.config(text="Servers: %d / %d" % (len(self.configs), MAX_SERVERS))
        headers = ["Server Name", "Capacity (req/s)", "Max Queue", "Initial Load (%)",
                   "Response Time (ms)", "Availability (%)", "Available"]
        for col, text in enumerate(headers):
            tk.Label(self.config_grid, text=text, font=(FONT, 10, "bold"), bg=CARD, fg=GRAY
                     ).grid(row=0, column=col, padx=6, pady=(0, 4), sticky="w")
        for r, c in enumerate(self.configs, start=1):
            vars_ = {
                "name": tk.StringVar(value=c.name),
                "capacity": tk.StringVar(value=str(c.capacity)),
                "max_queue": tk.StringVar(value=str(c.max_queue)),
                "initial_load": tk.StringVar(value=str(c.initial_load)),
                "response_time": tk.StringVar(value=str(c.response_time)),
                "availability": tk.StringVar(value=str(c.availability)),
                "up": tk.BooleanVar(value=c.up),
            }
            widths = {"name": 14, "capacity": 10, "max_queue": 8, "initial_load": 10,
                      "response_time": 10, "availability": 10}
            for col, key in enumerate(widths):
                tk.Entry(self.config_grid, textvariable=vars_[key], width=widths[key],
                         font=(FONT, 11)).grid(row=r, column=col, padx=6, pady=3, sticky="w")
            tk.Checkbutton(self.config_grid, variable=vars_["up"], bg=CARD, activebackground=CARD
                           ).grid(row=r, column=6, padx=6)
            self.config_vars.append(vars_)

    def read_config_rows(self):
        """Read the entry boxes. Raises ValueError with a friendly message if something is wrong."""
        new_configs = []
        limits = {"capacity": (10, 500), "max_queue": (1, 200), "initial_load": (0, 90),
                  "response_time": (1, 1000), "availability": (1, 100)}
        for row_number, v in enumerate(self.config_vars, start=1):
            name = v["name"].get().strip()
            if not name:
                raise ValueError("Server %d needs a name." % row_number)
            if len(name) > 20:
                raise ValueError("Server %d: the name must be 20 characters or fewer." % row_number)
            numbers = {}
            for key, (low, high) in limits.items():
                try:
                    number = int(float(v[key].get()))
                except (ValueError, OverflowError):
                    raise ValueError("Server %d: '%s' must be a number." % (row_number, key))
                if not low <= number <= high:
                    raise ValueError("Server %d: %s must be between %d and %d."
                                     % (row_number, key.replace("_", " "), low, high))
                numbers[key] = number
            new_configs.append(ServerConfig(name=name, up=bool(v["up"].get()), **numbers))
        # names appear on charts and cards, so every server needs its own name
        seen = set()
        for c in new_configs:
            if c.name.lower() in seen:
                raise ValueError("Every server needs a different name ('%s' is used twice)." % c.name)
            seen.add(c.name.lower())
        return new_configs

    def apply_config(self):
        if self.running:
            return
        try:
            new_configs = self.read_config_rows()
            new_weights = {}
            for key, var in self.weight_vars.items():
                try:
                    value = float(var.get())
                except ValueError:
                    raise ValueError("Weight '%s' must be a number." % key)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Weights must be ordinary numbers that are not negative.")
                new_weights[key] = value
            if sum(new_weights.values()) <= 0:
                raise ValueError("At least one weight must be greater than 0.")
        except ValueError as error:
            messagebox.showerror("Invalid configuration", str(error))
            return
        self.configs = new_configs
        self.weights = new_weights
        self.sim = None                              # settings changed -> old results are outdated
        self.status_var.set("Configuration applied.")
        self.refresh_dashboard()
        self.redraw_chart()

    def servers_changed(self, message):
        """Call after servers were added or removed: rebuild everything that shows servers."""
        self.sim = None                              # a finished run no longer matches the servers
        self.build_config_rows()
        self.refresh_dashboard()
        self.redraw_chart()
        self.status_var.set(message)

    def next_server_name(self):
        """'Server N' with the smallest free N (names must be unique)."""
        used = {c.name.lower() for c in self.configs}
        number = len(self.configs) + 1
        while ("server %d" % number) in used:
            number += 1
        return "Server %d" % number

    def add_server(self):
        if self.running:
            return
        if len(self.configs) >= MAX_SERVERS:
            messagebox.showinfo("Limit reached", "The maximum number of servers is %d." % MAX_SERVERS)
            return
        try:
            self.configs = self.read_config_rows()   # keep anything the user already typed
        except ValueError as error:
            messagebox.showerror("Invalid configuration", str(error))
            return
        self.configs.append(ServerConfig(self.next_server_name(),
                                         capacity=110, max_queue=20, initial_load=0,
                                         response_time=55, availability=97))
        self.servers_changed("Server added. Now %d servers." % len(self.configs))

    def remove_server(self):
        if self.running:
            return
        if len(self.configs) <= MIN_SERVERS:
            messagebox.showinfo("Limit reached", "At least %d servers are needed." % MIN_SERVERS)
            return
        try:
            self.configs = self.read_config_rows()
        except ValueError as error:
            messagebox.showerror("Invalid configuration", str(error))
            return
        removed = self.configs.pop()
        self.servers_changed("Removed %s. Now %d servers." % (removed.name, len(self.configs)))

    def restore_defaults(self):
        if self.running:
            return
        self.configs = default_configs()
        self.weights = dict(DEFAULT_WEIGHTS)
        for key, var in self.weight_vars.items():
            var.set(str(self.weights[key]))
        self.servers_changed("Default configuration restored.")

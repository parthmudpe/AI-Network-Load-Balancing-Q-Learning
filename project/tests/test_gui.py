"""
test_gui.py - Tests that click the real GUI buttons (through their functions).

Run with:   python -m tests.test_gui          (needs a screen / display)
On a computer without a screen (e.g. a server):   xvfb-run -a python -m tests.test_gui

If Tkinter or a display is not available, the tests are skipped.
"""
import sys

try:
    import tkinter as tk
    from gui import dashboard
    from gui.dashboard import App, MAX_SERVERS, MIN_SERVERS
    from gui.charts import CHART_NAMES
    from algorithms import ALGORITHM_NAMES, AI_NAME
except Exception as error:                     # no tkinter at all
    print("SKIPPED: Tkinter is not available (%s)" % error)
    sys.exit(0)

# ---------- helpers ----------
messages = []        # every message box the GUI tried to show: (kind, text)
dashboard.messagebox.showinfo = lambda title, text, **k: messages.append(("info", text))
dashboard.messagebox.showerror = lambda title, text, **k: messages.append(("error", text))
dashboard.messagebox.showwarning = lambda title, text, **k: messages.append(("warning", text))


def close(app):
    """Let queued redraws finish, then close the window (avoids harmless Tk 'after' warnings)."""
    app.update()
    app.destroy()


def new_app(requests="60", duration="1", traffic="Normal", algorithm=None):
    app = App()
    app.update()
    app.traffic_var.set(traffic)
    app.requests_var.set(requests)
    app.duration_var.set(duration)           # short runs keep the tests fast
    if algorithm:
        app.algorithm_var.set(algorithm)
    messages.clear()
    return app


def wait_until_finished(app, limit=600):
    for _ in range(limit * 10):
        if not app.running:
            return
        app.update()
        app.after(5)
    raise AssertionError("the simulation did not finish")


def check_consistent(app):
    """Cards, configuration rows and (if present) the simulator must all show the same servers."""
    n = len(app.configs)
    assert len(app.cards) == n, "cards %d != servers %d" % (len(app.cards), n)
    assert len(app.config_vars) == n, "config rows %d != servers %d" % (len(app.config_vars), n)
    if app.sim is not None:
        assert len(app.sim.servers) == n


def add_many(app, count):
    for _ in range(count):
        app.add_server()


# ---------- tests ----------
def test_add_servers_up_to_25_and_no_further():
    app = new_app()
    assert MAX_SERVERS == 25
    add_many(app, 21)                                  # 4 default + 21 = 25
    assert len(app.configs) == 25
    check_consistent(app)
    app.add_server()                                   # the 26th must be refused
    assert len(app.configs) == 25
    assert messages and messages[-1][0] == "info" and "25" in messages[-1][1]
    names = [c.name for c in app.configs]
    assert len(set(names)) == 25, "server names must be unique"
    close(app)


def test_remove_servers_down_to_minimum_and_no_further():
    app = new_app()
    add_many(app, 21)
    for _ in range(25 - MIN_SERVERS):
        app.remove_server()
    assert len(app.configs) == MIN_SERVERS
    check_consistent(app)
    messages.clear()
    app.remove_server()                                # below the minimum must be refused
    assert len(app.configs) == MIN_SERVERS
    assert messages and messages[-1][0] == "info"
    close(app)


def test_names_stay_unique_after_remove_and_add():
    app = new_app()
    app.config_vars[3]["name"].set("Server 5")         # the next free number is now taken
    app.add_server()
    names = [c.name for c in app.configs]
    assert len(set(n.lower() for n in names)) == len(names), names
    app.remove_server()
    app.remove_server()
    app.add_server()
    names = [c.name for c in app.configs]
    assert len(set(n.lower() for n in names)) == len(names), names
    close(app)


def test_unsaved_edits_survive_add_and_remove():
    app = new_app()
    app.config_vars[0]["capacity"].set("333")
    app.add_server()
    assert app.configs[0].capacity == 333
    assert app.config_vars[0]["capacity"].get() == "333"
    app.config_vars[0]["capacity"].set("222")
    app.remove_server()
    assert app.configs[0].capacity == 222
    close(app)


def test_duplicate_names_are_rejected():
    app = new_app()
    app.config_vars[1]["name"].set(app.config_vars[0]["name"].get())
    before = [c.name for c in app.configs]
    app.apply_config()
    assert [c.name for c in app.configs] == before, "bad configuration must not be applied"
    assert messages and messages[-1][0] == "error" and "different name" in messages[-1][1]
    close(app)


def test_bad_numbers_are_rejected_without_crashing():
    app = new_app()
    for text in ("abc", "inf", "nan", "-5", "99999"):
        messages.clear()
        app.config_vars[0]["capacity"].set(text)
        app.apply_config()
        assert messages and messages[-1][0] == "error", "capacity=%r was not rejected" % text
    app.config_vars[0]["capacity"].set("100")
    for text in ("abc", "nan", "inf", "-1"):
        messages.clear()
        app.weight_vars["load"].set(text)
        app.apply_config()
        assert messages and messages[-1][0] == "error", "weight=%r was not rejected" % text
    app.weight_vars["load"].set("0.25")
    for text in ("inf", "nan", "", "0", "-3"):         # invalid simulation settings
        messages.clear()
        app.requests_var.set(text)
        app.start_simulation()
        assert messages and messages[-1][0] == "error" and not app.running, "requests=%r" % text
    close(app)


def test_editing_buttons_are_locked_while_running():
    app = new_app(duration="2")
    app.start_simulation()
    app.update()
    assert app.running
    for button in app.config_buttons + [app.apply_button, app.compare_button,
                                        app.start_button, app.demo_button]:
        assert str(button["state"]) == "disabled"
    count = len(app.configs)
    sim = app.sim
    app.add_server()
    app.remove_server()
    app.restore_defaults()
    app.apply_config()
    assert len(app.configs) == count and app.sim is sim and app.running, \
        "changing servers during a run must be ignored"
    wait_until_finished(app)
    for button in app.config_buttons + [app.apply_button, app.compare_button, app.start_button]:
        assert str(button["state"]) == "normal"
    close(app)


def test_run_after_adding_and_removing_servers():
    app = new_app()
    add_many(app, 6)                                   # 10 servers (3-column cards)
    app.start_simulation()
    wait_until_finished(app)
    check_consistent(app)
    s = app.sim.summary()
    assert len(s["names"]) == 10 and s["processed"] + s["dropped"] == s["total"]
    app.remove_server()
    assert app.sim is None, "old results must be cleared when servers change"
    check_consistent(app)
    app.start_simulation()
    wait_until_finished(app)
    assert len(app.sim.summary()["names"]) == 9
    close(app)


def test_fail_and_recover_during_a_run():
    app = new_app(requests="200", duration="2")
    app.start_simulation()
    for _ in range(5):
        app.update()
        app.after(5)
    app.toggle_server(0)                               # fail
    assert app.sim.servers[0].available is False
    app.refresh_dashboard()
    assert app.cards[0]["badge"]["text"] == "FAILED"
    app.toggle_server(0)                               # recover
    assert app.sim.servers[0].available is True
    wait_until_finished(app)
    s = app.sim.summary()
    assert s["processed"] + s["dropped"] == s["total"]
    close(app)


def test_fail_and_recover_when_idle_keeps_config_tab_in_step():
    app = new_app()
    app.config_vars[2]["capacity"].set("140")          # an unsaved edit must not be lost
    app.toggle_server(1)
    assert app.configs[1].up is False and app.config_vars[1]["up"].get() is False
    assert app.config_vars[2]["capacity"].get() == "140"
    app.toggle_server(1)
    assert app.configs[1].up is True and app.config_vars[1]["up"].get() is True
    close(app)


def test_ai_runs_from_the_dashboard():
    app = new_app(requests="200", duration="2", algorithm=AI_NAME)
    assert app.q_table is None
    app.start_simulation()                             # trains the AI first, then runs it
    assert app.q_table, "the AI should have been trained"
    first_table = app.q_table
    wait_until_finished(app)
    s = app.sim.summary()
    assert app.sim.algorithm.name == AI_NAME
    assert s["processed"] + s["dropped"] == s["total"] and s["processed"] > 0
    assert app.cards[0]["score"]["text"].startswith("Q-value")
    # same scenario again -> no retraining; changed scenario -> retraining
    app.start_simulation()
    assert app.q_table is first_table
    wait_until_finished(app)
    app.config_vars[0]["capacity"].set("200")
    app.apply_config()
    app.start_simulation()
    assert app.q_table is not first_table, "changed servers should trigger new training"
    wait_until_finished(app)
    close(app)


def test_comparison_has_all_six_algorithms_and_the_ai():
    app = new_app(requests="300", duration="2", traffic="High")
    app.run_comparison()
    rows = [app.compare_table.item(r)["values"] for r in app.compare_table.get_children()]
    assert len(rows) == 6 and len(app.comparison) == 6
    assert [r[0] for r in rows] == ALGORITHM_NAMES
    assert all(len(r) == 6 for r in rows)
    note = app.compare_note["text"]
    assert "Q-Learning (AI)" in note and "rank" in note
    # the table rows are visible without scrolling
    assert int(app.compare_table["height"]) >= 6
    for r in app.comparison:
        s = r["summary"]
        assert s["processed"] + s["dropped"] == s["total"] == 300
    close(app)


def test_comparison_and_all_charts_with_25_servers():
    app = new_app(requests="1500", duration="2", traffic="Burst")
    add_many(app, 21)
    app.run_comparison()
    assert len(app.compare_table.get_children()) == 6
    app.start_simulation()
    wait_until_finished(app)
    for name in CHART_NAMES:
        app.chart_var.set(name)
        app.redraw_chart()                             # must not raise
        axes = app.chart_figure.axes[0]
        if name in ("Requests Handled by Each Server", "Server Utilization",
                    "Average Response Time", "Load Distribution"):
            assert len(axes.patches) == 25, "%s drew %d bars" % (name, len(axes.patches))
        if name == "Queue Length":
            assert len(axes.lines) == 25
    assert len(app.compare_figure.axes[0].patches) == 6
    close(app)


def test_charts_with_duplicate_looking_names_show_every_server():
    app = new_app()
    app.start_simulation()
    wait_until_finished(app)
    app.sim.servers[1].name = app.sim.servers[0].name      # two servers with the same label
    app.sim.metrics.server_names[1] = app.sim.metrics.server_names[0]
    app.chart_var.set("Requests Handled by Each Server")
    app.redraw_chart()
    xs = sorted(round(p.get_x() + p.get_width() / 2) for p in app.chart_figure.axes[0].patches)
    assert xs == [0, 1, 2, 3], "every server needs its own bar position: %s" % xs
    close(app)


def test_demo_is_slow_but_next_normal_run_is_not():
    app = new_app()
    app.run_demo()
    assert app.steps_per_frame == 1 and app.running
    wait_until_finished(app)
    app.start_simulation()
    assert app.steps_per_frame == 5, "a normal run after the demo must use normal speed"
    wait_until_finished(app)
    close(app)


def test_restore_defaults_resets_servers_and_weights():
    app = new_app()
    add_many(app, 5)
    app.weight_vars["load"].set("0.9")
    app.restore_defaults()
    assert len(app.configs) == 4 and app.weight_vars["load"].get() == "0.25"
    check_consistent(app)
    close(app)


def test_reset_button_works_at_any_time():
    app = new_app(duration="3")
    app.start_simulation()
    app.update()
    app.reset_simulation()
    assert not app.running and app.sim is None
    assert str(app.start_button["state"]) == "normal"
    for button in app.config_buttons:
        assert str(button["state"]) == "normal"
    close(app)


if __name__ == "__main__":
    try:
        probe = tk.Tk()
        probe.destroy()
    except tk.TclError as error:
        print("SKIPPED: no display available (%s)\nTip: xvfb-run -a python -m tests.test_gui" % error)
        sys.exit(0)

    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for t in tests:
        print("Running %s ..." % t.__name__)
        t()
        print("  PASSED")
    print("\nAll %d GUI tests passed." % len(tests))

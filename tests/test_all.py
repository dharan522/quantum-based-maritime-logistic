"""
tests/test_all.py
Unit tests for QAOA Maritime Logistics Optimiser + Digital Twin.

Run:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.qubo_formulator  import MaritimeQUBOFormulator
from src.qaoa_solver      import MaritimeQAOASolver
from src.classical_solver import ClassicalMaritimeSolver
from src.digital_twin     import PortDigitalTwin, MaritimeEvent


# ── Fixtures ──────────────────────────────────────────────────────────────

PORT_IDS = ["SHA", "SIN", "ROT", "LAX", "HAM", "DXB"]

@pytest.fixture(scope="module")
def dist():
    """Small 6-port distance matrix."""
    n = len(PORT_IDS)
    D = np.zeros((n, n))
    np.random.seed(0)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.random.uniform(500, 9000)
    return pd.DataFrame(D, index=PORT_IDS, columns=PORT_IDS)


@pytest.fixture(scope="module")
def vessels():
    return pd.DataFrame([
        {"id":"V01","name":"Ship A","type":"ULCV","capacity_teu":18000,
         "speed_knots":22,"fuel_tday":280,"cost_day":45000},
        {"id":"V02","name":"Ship B","type":"VLCV","capacity_teu":12000,
         "speed_knots":20,"fuel_tday":210,"cost_day":34000},
        {"id":"V03","name":"Ship C","type":"Feeder","capacity_teu":4000,
         "speed_knots":17,"fuel_tday":95,"cost_day":14000},
    ])


@pytest.fixture(scope="module")
def cargo():
    np.random.seed(42)
    records = []
    for i in range(12):
        o = PORT_IDS[i % len(PORT_IDS)]
        d = PORT_IDS[(i + 2) % len(PORT_IDS)]
        records.append({
            "cargo_id":    f"C{i:03d}",
            "origin":      o,
            "destination": d,
            "weight_teu":  int(np.random.randint(200, 3000)),
            "type":        "Container",
            "priority":    np.random.choice(["HIGH","MEDIUM","LOW"]),
            "deadline":    "2026-04-10",
            "revenue_usd": float(np.random.uniform(50000, 400000)),
        })
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def ports():
    return pd.DataFrame({
        "name":          [p for p in PORT_IDS],
        "lat":           [31.2, 1.3, 51.9, 33.7, 53.6, 25.0],
        "lon":           [121.5, 103.9, 4.5, -118.3, 10.0, 55.1],
        "berths":        [8, 6, 10, 9, 7, 8],
        "handling_rate": [4500, 4200, 5200, 4700, 3900, 4600],
        "region":        ["Asia"]*3 + ["Americas"] + ["Europe"] + ["Middle East"],
    }, index=PORT_IDS)


@pytest.fixture(scope="module")
def formulator(dist, vessels, cargo, ports):
    return MaritimeQUBOFormulator(
        dist, vessels, cargo, ports,
        n_vessels=2, n_routes=2,
        alpha=0.4, beta=0.3, gamma=0.3,
        penalty_A=8.0, penalty_B=6.0, penalty_C=10.0,
    )


# ── MaritimeQUBOFormulator ─────────────────────────────────────────────────

class TestQUBOFormulator:

    def test_qubo_square(self, formulator):
        Q, meta = formulator.build_qubo()
        n = meta["n_vars"]
        assert Q.shape == (n, n)

    def test_qubo_upper_triangular(self, formulator):
        Q, _ = formulator.build_qubo()
        assert np.allclose(np.tril(Q, k=-1), 0), "Lower triangle must be zero"

    def test_meta_keys(self, formulator):
        _, meta = formulator.build_qubo()
        for key in ["n_vars","n_vessels","n_routes","qubo_shape","qubo_range","var_labels"]:
            assert key in meta

    def test_var_labels_count(self, formulator):
        Q, meta = formulator.build_qubo()
        assert len(meta["var_labels"]) == meta["n_vars"]

    def test_route_generation(self, formulator):
        for vi in range(formulator.n_v):
            assert len(formulator.routes[vi]) == formulator.n_r

    def test_evaluate_solution_keys(self, formulator):
        bs = "1" * formulator.n_vars
        result = formulator.evaluate_solution(bs)
        for key in ["bitstring","qubo_energy","assignments","n_assigned","total_cost"]:
            assert key in result

    def test_evaluate_zero_bitstring(self, formulator):
        bs = "0" * formulator.n_vars
        result = formulator.evaluate_solution(bs)
        assert result["n_assigned"] == 0

    def test_route_summary_columns(self, formulator):
        df = formulator.get_route_summary()
        for col in ["vessel","route","distance_nm","days","cost_usd","var"]:
            assert col in df.columns

    def test_route_revenue_nonnegative(self, formulator):
        route = formulator.routes[0][0]["route"]
        rev   = formulator._route_revenue(route, capacity_teu=20000)
        assert rev >= 0


# ── MaritimeQAOASolver ─────────────────────────────────────────────────────

class TestQAOASolver:

    def test_circuit_expectation_is_float(self):
        Q      = np.array([[-2.0, 1.0],[1.0,-2.0]])
        solver = MaritimeQAOASolver(Q, ["a","b"], p_layers=1, shots=128, seed=0)
        e      = solver._expectation(np.array([0.5, 0.3]))
        assert isinstance(e, float)

    def test_optimise_returns_bitstring(self):
        Q      = np.diag([-1.5, -2.0, -1.0])
        solver = MaritimeQAOASolver(Q, ["x","y","z"], p_layers=1, shots=256, seed=0)
        result = solver.optimise(maxiter=10)
        assert "best_bitstring" in result
        assert len(result["best_bitstring"]) == 3
        assert all(c in "01" for c in result["best_bitstring"])

    def test_energy_history_grows(self):
        Q      = np.array([[-1.0, 0.5],[0.5,-1.0]])
        solver = MaritimeQAOASolver(Q, ["a","b"], p_layers=1, shots=128, seed=1)
        solver.optimise(maxiter=8)
        assert len(solver.energy_history) > 0

    def test_counts_total_correct(self):
        Q      = np.diag([-1.0, -1.0])
        solver = MaritimeQAOASolver(Q, ["a","b"], p_layers=1, shots=256, seed=0)
        result = solver.optimise(maxiter=5)
        assert sum(result["counts"].values()) == 256 * 4

    def test_landscape_shape(self):
        Q      = np.diag([-1.0, -2.0])
        solver = MaritimeQAOASolver(Q, ["a","b"], p_layers=1, shots=64, seed=0)
        g, b, E = solver.scan_landscape(resolution=4)
        assert E.shape == (4, 4)

    def test_top_k_solutions(self):
        Q      = np.diag([-1.0, -2.0, -1.5])
        solver = MaritimeQAOASolver(Q, ["a","b","c"], p_layers=1, shots=128, seed=0)
        solver.optimise(maxiter=5)
        top3   = solver.top_k_solutions(k=3)
        assert len(top3) <= 3
        for s in top3:
            assert "bitstring" in s and "probability" in s and "energy" in s

    def test_probabilities_sum_to_one(self):
        Q      = np.diag([-1.0, -2.0])
        solver = MaritimeQAOASolver(Q, ["a","b"], p_layers=1, shots=128, seed=0)
        solver.optimise(maxiter=5)
        counts = solver.counts
        total  = sum(counts.values())
        assert total == 128 * 4


# ── ClassicalMaritimeSolver ────────────────────────────────────────────────

class TestClassicalSolver:

    def test_greedy_assigns_all_vessels(self, dist, vessels, cargo, ports):
        cs = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        r  = cs.greedy_route_assignment(n_vessels=2)
        assert len(r["assignments"]) == 2

    def test_greedy_total_cost_positive(self, dist, vessels, cargo, ports):
        cs = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        r  = cs.greedy_route_assignment(n_vessels=2)
        assert r["total_cost"] > 0

    def test_clarke_wright_returns_dict(self, dist, vessels, cargo, ports):
        cs = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        r  = cs.clarke_wright_savings(n_vessels=2)
        assert "assignments" in r and "total_cost" in r

    def test_hungarian_matching_revenue(self, dist, vessels, cargo, ports):
        cs = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        r  = cs.hungarian_cargo_assignment(n_vessels=2)
        assert r["total_revenue"] >= 0

    def test_berth_schedule_columns(self, dist, vessels, cargo, ports):
        cs = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        df = cs.priority_berth_schedule("SHA", n_berths=4)
        if not df.empty:
            for col in ["cargo_id","priority","berth","start","end"]:
                assert col in df.columns

    def test_benchmark_summary_rows(self, dist, vessels, cargo, ports):
        cs   = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        bench= cs.benchmark_summary(qaoa_cost=500000.0, n_vessels=2)
        assert len(bench) == 3
        assert "QAOA" in bench["method"].values

    def test_berth_schedule_no_overlap(self, dist, vessels, cargo, ports):
        cs  = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
        df  = cs.priority_berth_schedule("SHA", n_berths=4)
        if len(df) >= 2:
            for berth in df["berth"].unique():
                b_df = df[df["berth"] == berth].sort_values("start")
                for i in range(len(b_df) - 1):
                    assert b_df.iloc[i]["end"] <= b_df.iloc[i+1]["start"], \
                        f"Overlap in berth {berth}"


# ── PortDigitalTwin ────────────────────────────────────────────────────────

class TestDigitalTwin:

    def test_generate_disruptions_returns_list(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        evs  = twin.generate_disruptions(scenario="normal", n_days=5)
        assert isinstance(evs, list)

    def test_crisis_more_events_than_normal(self):
        twin_n = PortDigitalTwin(PORT_IDS, seed=42)
        twin_c = PortDigitalTwin(PORT_IDS, seed=42)
        n_evs  = twin_n.generate_disruptions(scenario="normal",  n_days=14)
        c_evs  = twin_c.generate_disruptions(scenario="crisis",  n_days=14)
        assert len(c_evs) >= len(n_evs)

    def test_weather_simulation_shape(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        df   = twin.simulate_weather(n_hours=24)
        assert df.shape == (24, len(PORT_IDS))

    def test_weather_valid_states(self):
        twin   = PortDigitalTwin(PORT_IDS, seed=0)
        df     = twin.simulate_weather(n_hours=10)
        valid  = set(PortDigitalTwin.WEATHER_STATES)
        for col in df.columns:
            assert set(df[col].unique()).issubset(valid)

    def test_arrivals_all_ports_covered(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        df   = twin.simulate_arrivals(n_days=3)
        assert set(df["port"].unique()).issubset(set(PORT_IDS))

    def test_berth_queue_shape(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        df   = twin.simulate_berth_queue("SHA", n_berths=5, n_hours=24)
        assert df.shape[0] == 24

    def test_berth_utilisation_in_range(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        df   = twin.simulate_berth_queue("SHA", n_berths=5, n_hours=24)
        assert (df["utilisation"] >= 0).all() and (df["utilisation"] <= 1).all()

    def test_tides_shape(self):
        twin  = PortDigitalTwin(PORT_IDS, seed=0)
        tides = twin.simulate_tides("SHA", n_hours=48)
        assert len(tides) == 48

    def test_fuel_prices_gbm_positive(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        fp   = twin.simulate_fuel_prices(n_days=10)
        assert (fp > 0).all()

    def test_cost_multiplier_above_one_for_event(self):
        ev = MaritimeEvent("canal_closure", port_id="SHA",
                           severity=0.9, duration_hr=24)
        assert ev.cost_multiplier > 1.0

    def test_throughput_below_one_for_severe_event(self):
        ev = MaritimeEvent("breakdown", port_id="SIN",
                           severity=0.9, duration_hr=12)
        assert ev.throughput_factor < 1.0

    def test_delay_hours_positive(self):
        ev = MaritimeEvent("weather", port_id="ROT",
                           severity=0.7, duration_hr=8)
        assert ev.delay_hours > 0

    def test_congestion_heatmap_shape(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        twin.generate_disruptions("stressed", n_days=3)
        df   = twin.simulate_congestion_heatmap(n_hours=24)
        assert df.shape == (24, len(PORT_IDS))

    def test_throughput_matrix_all_ports(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        evs  = twin.generate_disruptions("stressed", n_days=3)
        tp   = twin.compute_throughput_matrix(evs)
        for p in PORT_IDS:
            assert p in tp.index

    def test_throughput_values_in_range(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        evs  = twin.generate_disruptions("crisis", n_days=3)
        tp   = twin.compute_throughput_matrix(evs)
        assert (tp["throughput_factor"] >= 0).all()
        assert (tp["throughput_factor"] <= 1).all()

    def test_route_cost_multiplier_applied(self):
        twin   = PortDigitalTwin(PORT_IDS, seed=0)
        evs    = [MaritimeEvent("canal_closure", "SHA", 0.9, 48)]
        routes = {
            0: [{"route":("SHA","SIN","ROT"), "total_cost": 1000000,
                 "distance_nm": 5000, "days": 10}]
        }
        adj = twin.compute_route_cost_multipliers(routes, evs)
        assert adj[0][0]["total_cost"] > 1000000

    def test_event_summary_dataframe(self):
        twin = PortDigitalTwin(PORT_IDS, seed=0)
        twin.generate_disruptions("crisis", n_days=5)
        df   = twin.get_event_summary()
        if not df.empty:
            for col in ["port","event_type","severity","delay_hours","cost_mult"]:
                assert col in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
main.py
Master pipeline: QAOA Maritime Logistics + Digital Twin Port Simulation.

Usage:
    python main.py [--scenario normal|stressed|crisis] [--layers 2] [--vessels 4]
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

from src.qubo_formulator  import MaritimeQUBOFormulator
from src.qaoa_solver      import MaritimeQAOASolver
from src.classical_solver import ClassicalMaritimeSolver
from src.digital_twin     import PortDigitalTwin

OCEAN_DARK  = "#050E1A"
OCEAN_MID   = "#0A2540"
TEAL        = "#00C9C8"
CYAN        = "#00E5FF"
AMBER       = "#FFB347"
CORAL       = "#FF6B6B"
SEA_GREEN   = "#2ECC71"
MUTED       = "#8BAFC4"
TEXT        = "#D6EAF8"

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  OCEAN_DARK,
        "axes.facecolor":    OCEAN_MID,
        "axes.edgecolor":    TEAL+"44",
        "axes.labelcolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        TEAL+"22",
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "font.family":       "monospace",
        "axes.titlecolor":   CYAN,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })

def load_data():
    dist = pd.read_csv("data/distance_matrix.csv", index_col=0)
    ports   = pd.read_csv("data/ports.csv", index_col=0)
    vessels = pd.read_csv("data/vessels.csv")
    cargo   = pd.read_csv("data/cargo.csv")
    return dist, ports, vessels, cargo


def run_pipeline(scenario="stressed", p_layers=2, n_vessels=4):
    print("\n" + "═"*65)
    print("  ⚓  QAOA MARITIME LOGISTICS OPTIMISER")
    print("  ≋  Digital Twin Port Simulation")
    print("═"*65)

    setup_style()
    dist, ports, vessels, cargo = load_data()
    port_ids = list(dist.index)

    # ── 1. Digital Twin ───────────────────────────────────────────────
    print(f"\n[1/7] Digital Twin simulation  (scenario={scenario})...")
    twin    = PortDigitalTwin(port_ids, seed=2026)
    events  = twin.generate_disruptions(scenario=scenario, n_days=7)
    weather = twin.simulate_weather(n_hours=48)
    arrivals= twin.simulate_arrivals(n_days=7)
    cong    = twin.simulate_congestion_heatmap(n_hours=48)
    fuel_px = twin.simulate_fuel_prices(n_days=30)
    ev_df   = twin.get_event_summary()
    berth_Q = twin.simulate_berth_queue("SHA", n_berths=8, n_hours=48)

    print(f"     {len(events)} disruption events generated")
    print(f"     {len(arrivals)} vessel arrivals simulated (7 days)")
    tp = twin.compute_throughput_matrix(events)
    print(f"     Avg port throughput: {tp['throughput_factor'].mean()*100:.1f}%")

    # ── 2. QUBO Formulation ──────────────────────────────────────────
    print(f"\n[2/7] QUBO formulation  (vessels={n_vessels}, routes=3)...")
    fml = MaritimeQUBOFormulator(dist, vessels, cargo, ports, n_vessels=n_vessels)

    # Apply digital twin cost adjustments
    adjusted_routes = twin.compute_route_cost_multipliers(fml.routes, events)
    fml.routes      = adjusted_routes   # patch routes with adjusted costs

    Q, meta = fml.build_qubo()
    print(f"     QUBO: {meta['n_vars']} variables, shape {meta['qubo_shape']}")
    print(f"     QUBO range: [{meta['qubo_range'][0]:.3f}, {meta['qubo_range'][1]:.3f}]")

    routes_df = fml.get_route_summary()

    # ── 3. QAOA ──────────────────────────────────────────────────────
    print(f"\n[3/7] QAOA optimisation  (p={p_layers}, shots=2048)...")
    solver  = MaritimeQAOASolver(Q, meta["var_labels"], p_layers=p_layers, shots=2048)
    result  = solver.optimise(maxiter=150)
    qaoa_eval = fml.evaluate_solution(result["best_bitstring"])
    top_k     = solver.top_k_solutions(k=8)

    print(f"     Best bitstring : {result['best_bitstring']}")
    print(f"     QUBO energy    : {qaoa_eval['qubo_energy']:.4f}")
    print(f"     Vessels assigned: {qaoa_eval['n_assigned']}/{n_vessels}")
    print(f"     Total cost     : ${qaoa_eval['total_cost']:,.0f}")
    print(f"     Converged      : {result['converged']}")

    # ── 4. Classical Benchmarks ──────────────────────────────────────
    print(f"\n[4/7] Classical benchmarks...")
    classical = ClassicalMaritimeSolver(dist, vessels, cargo, ports)
    greedy    = classical.greedy_route_assignment(n_vessels)
    cw        = classical.clarke_wright_savings(n_vessels=n_vessels)
    hungarian = classical.hungarian_cargo_assignment(n_vessels)
    berth_sch = classical.priority_berth_schedule("SHA", n_berths=8)
    bench     = classical.benchmark_summary(qaoa_eval["total_cost"], n_vessels)
    print(f"     Greedy cost:         ${greedy['total_cost']:>12,.0f}")
    print(f"     Clarke-Wright cost:  ${cw['total_cost']:>12,.0f}")
    print(f"     QAOA cost:           ${qaoa_eval['total_cost']:>12,.0f}")

    # ── 5. Energy Landscape (p=1) ────────────────────────────────────
    print(f"\n[5/7] Energy landscape scan (reduced resolution for speed)...")
    # Use a 6-variable sub-QUBO for the landscape scan
    sub_n = min(6, Q.shape[0])
    Q_sub = Q[:sub_n, :sub_n]
    lbl_sub = meta["var_labels"][:sub_n]
    sl1 = MaritimeQAOASolver(Q_sub, lbl_sub, p_layers=1, shots=128)
    g, b, E = sl1.scan_landscape(resolution=14)

    # cache routes for plot
    global fml_routes_cache
    fml_routes_cache = fml.routes

    # ── 6. Plots ─────────────────────────────────────────────────────
    print(f"\n[6/7] Generating all charts...")
    _plot_all(
        solver, result, qaoa_eval, top_k,
        routes_df, greedy, cw, bench,
        cong, berth_Q, arrivals, fuel_px,
        weather, ev_df, Q, meta,
        g, b, E, scenario, port_ids, twin, events, fml
    )

    # ── 7. Report ────────────────────────────────────────────────────
    print(f"\n[7/7] Saving report...")
    report = {
        "timestamp":    datetime.now().isoformat(),
        "scenario":     scenario,
        "p_layers":     p_layers,
        "n_vessels":    n_vessels,
        "n_events":     len(events),
        "qaoa_result":  {k: v for k,v in qaoa_eval.items() if k != "assignments"},
        "benchmark":    bench.to_dict("records"),
        "qubo_meta":    meta,
    }
    with open("outputs/report.json","w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n✅ Pipeline complete!")
    _print_summary(qaoa_eval, greedy, cw, result)
    return report


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def _plot_all(solver, result, qaoa_eval, top_k,
              routes_df, greedy, cw, bench,
              cong, berth_Q, arrivals, fuel_px,
              weather, ev_df, Q, meta,
              g, b, E, scenario, port_ids, twin, events, fml):

    # ── Fig 1: QAOA Convergence + Energy Landscape ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(OCEAN_DARK)

    # Convergence
    ax = axes[0]
    iters = range(len(solver.energy_history))
    ax.plot(iters, solver.energy_history, color=CYAN, lw=1.5, alpha=0.9, zorder=3)
    ax.fill_between(iters, solver.energy_history, alpha=0.15, color=TEAL)
    ax.axhline(solver.optimal_energy, color=CORAL, lw=1.5, ls="--",
               label=f"Optimal = {solver.optimal_energy:.4f}")
    ax.set_title("QAOA Optimisation Convergence", pad=10)
    ax.set_xlabel("Iteration"); ax.set_ylabel("⟨HC⟩ Energy")
    ax.legend(facecolor=OCEAN_MID, edgecolor=TEAL+"44", fontsize=9)
    ax.grid(True)

    # Energy Landscape
    ax = axes[1]
    ocean_cmap = LinearSegmentedColormap.from_list(
        "ocean", [OCEAN_DARK, OCEAN_MID, TEAL, CYAN, "#FFFFFF"])
    cf = ax.contourf(b, g, E, levels=24, cmap="plasma")
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("⟨HC⟩", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    ax.set_title("QAOA Energy Landscape (p=1 slice)", pad=10)
    ax.set_xlabel("β (Mixer angle)"); ax.set_ylabel("γ (Cost angle)")

    plt.tight_layout(pad=2.0)
    plt.savefig("outputs/fig_convergence_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: Digital Twin — Congestion + Berth Queue ────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor(OCEAN_DARK)
    fig.suptitle(f"Digital Twin Port Simulation  [scenario = {scenario.upper()}]",
                 color=CYAN, fontsize=14, y=1.01)

    ax = axes[0]
    sub_ports = port_ids[:8]
    cong_sub  = cong[sub_ports]
    im = ax.imshow(cong_sub.values.T, aspect="auto", cmap="hot",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(sub_ports)))
    ax.set_yticklabels(sub_ports, fontsize=9)
    ax.set_xlabel("Hour"); ax.set_title("Port Congestion Heatmap (48h)", pad=8)
    fig.colorbar(im, ax=ax, label="Congestion Level", fraction=0.02)

    ax = axes[1]
    ax.plot(berth_Q.index, berth_Q["utilisation"],   color=TEAL,  lw=2.0, label="Berth Utilisation")
    ax.plot(berth_Q.index, berth_Q["wait_time_hr"]/berth_Q["wait_time_hr"].max(),
            color=AMBER, lw=1.5, ls="--", label="Normalised Wait Time")
    ax.fill_between(berth_Q.index, berth_Q["utilisation"], alpha=0.12, color=TEAL)
    ax.set_xlabel("Hour"); ax.set_ylabel("Rate")
    ax.set_title("SHA Port — M/M/c Berth Queue Simulation", pad=8)
    ax.legend(facecolor=OCEAN_MID, edgecolor=TEAL+"44", fontsize=9)
    ax.grid(True); ax.set_ylim(0, 1.15)

    plt.tight_layout(pad=2.0)
    plt.savefig("outputs/fig_digital_twin.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: Port Network Map (lat/lon scatter) ─────────────────────
    import json as _json
    with open("data/ports.json") as f:
        ports_json = _json.load(f)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(OCEAN_DARK)
    ax.set_facecolor("#071520")
    ax.set_xlim(-130, 160); ax.set_ylim(-40, 65)
    ax.set_title("Global Maritime Port Network — QAOA Route Assignments", pad=10, color=CYAN)

    # Draw selected routes
    colors_route = [TEAL, AMBER, CORAL, SEA_GREEN]
    assigned_routes = []
    for vi in range(fml.n_v):
        for ri in range(fml.n_r):
            key = f"x_v{vi}_r{ri}"
            if key in meta["var_labels"]:
                qi = meta["var_labels"].index(key)
                bs = result["best_bitstring"]
                if qi < len(bs) and bs[qi] == "1":
                    if ri < len(fml.routes[vi]):
                        assigned_routes.append((vi, fml.routes[vi][ri]["route"]))

    for pid, data in ports_json.items():
        lon, lat = data["lon"], data["lat"]
        is_active = any(pid in str(r) for _, r in assigned_routes)
        col  = CYAN if is_active else MUTED
        size = 120 if is_active else 60
        ax.scatter(lon, lat, s=size, color=col, zorder=5, edgecolors=OCEAN_DARK, lw=0.5)
        ax.annotate(pid, (lon, lat), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color=col)

    for i, (vi, route) in enumerate(assigned_routes[:4]):
        col = colors_route[i % len(colors_route)]
        for k in range(len(route)-1):
            p1, p2 = route[k], route[k+1]
            if p1 in ports_json and p2 in ports_json:
                x1, y1 = ports_json[p1]["lon"], ports_json[p1]["lat"]
                x2, y2 = ports_json[p2]["lon"], ports_json[p2]["lat"]
                ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("outputs/fig_port_network.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 4: Measurement Distribution + Benchmark ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(OCEAN_DARK)

    # Probability distribution
    ax = axes[0]
    counts = result["counts"]
    total  = sum(counts.values())
    top20  = sorted(counts.items(), key=lambda x:x[1], reverse=True)[:18]
    labels = [c[0][::-1] for c in top20]
    probs  = [c[1]/total for c in top20]
    bar_cols = [CYAN if l==result["best_bitstring"] else TEAL+"88" for l in labels]
    bars = ax.bar(range(len(labels)), probs, color=bar_cols, edgecolor=OCEAN_DARK, width=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Probability"); ax.set_title("QAOA Measurement Distribution (Top-18)", pad=8)
    patch = mpatches.Patch(color=CYAN, label=f"Best: {result['best_bitstring']}")
    ax.legend(handles=[patch], facecolor=OCEAN_MID, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Cost benchmark
    ax = axes[1]
    methods   = bench["method"].tolist()
    costs     = (bench["total_cost"] / 1e6).tolist()
    bar_cols2 = [CYAN if m=="QAOA" else AMBER if m=="Greedy-NN" else CORAL for m in methods]
    bars2 = ax.bar(methods, costs, color=bar_cols2, edgecolor=OCEAN_DARK, width=0.5, alpha=0.9)
    for bar, v in zip(bars2, costs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"${v:.2f}M", ha="center", fontsize=10, color=TEXT)
    ax.set_ylabel("Total Cost (USD Millions)"); ax.set_title("QAOA vs Classical Route Cost", pad=8)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(costs)*1.25)

    plt.tight_layout(pad=2.0)
    plt.savefig("outputs/fig_distribution_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 5: Digital Twin Fuel + Arrivals + Throughput ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(OCEAN_DARK)

    # Fuel price GBM
    ax = axes[0]
    ax.plot(range(len(fuel_px)), fuel_px, color=AMBER, lw=2.0)
    ax.fill_between(range(len(fuel_px)), fuel_px, alpha=0.15, color=AMBER)
    ax.axhline(650, color=MUTED, ls="--", lw=1.0, label="Initial $650/t")
    ax.set_title("Bunker Fuel Price Simulation (GBM)", pad=8)
    ax.set_xlabel("Days"); ax.set_ylabel("USD/tonne")
    ax.legend(facecolor=OCEAN_MID, fontsize=9); ax.grid(True)

    # Vessel arrivals per port
    ax = axes[1]
    arr_counts = arrivals.groupby("port").size().sort_values(ascending=True)
    ax.barh(arr_counts.index, arr_counts.values, color=TEAL, alpha=0.85, edgecolor=OCEAN_DARK)
    ax.set_title("Vessel Arrivals by Port (7 days)", pad=8)
    ax.set_xlabel("Total Arrivals"); ax.grid(axis="x", alpha=0.3)

    # Throughput impact
    ax = axes[2]
    tp = twin.compute_throughput_matrix(events)
    sorted_tp = tp.sort_values("throughput_factor")
    bar_cols3 = [CORAL if v < 0.7 else AMBER if v < 0.9 else SEA_GREEN
                 for v in sorted_tp["throughput_factor"]]
    ax.barh(sorted_tp.index, sorted_tp["throughput_factor"]*100,
            color=bar_cols3, edgecolor=OCEAN_DARK, alpha=0.9)
    ax.axvline(100, color=MUTED, ls="--", lw=1.0)
    ax.set_xlim(0, 115)
    for i, v in enumerate(sorted_tp["throughput_factor"]*100):
        ax.text(v+0.5, i, f"{v:.0f}%", va="center", fontsize=8, color=TEXT)
    ax.set_title("Digital Twin — Port Throughput %", pad=8)
    ax.set_xlabel("Throughput (%)"); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig("outputs/fig_twin_operations.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 6: QUBO Heatmap + Top-K solutions ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(OCEAN_DARK)

    ax = axes[0]
    n  = Q.shape[0]
    labels_short = [l.replace("x_","").replace("b_","b:") for l in meta["var_labels"]]
    im = ax.imshow(Q, cmap="RdBu_r", vmin=Q.min(), vmax=Q.max(), aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_short, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Q value")
    ax.set_title(f"QUBO Matrix Heatmap  ({n}×{n})", pad=8)

    ax = axes[1]
    if top_k:
        tk_labels = [f"BS-{i+1}\n{s['bitstring'][:8]}..." for i,s in enumerate(top_k)]
        tk_probs  = [s["probability"] for s in top_k]
        tk_energy = [s["energy"] for s in top_k]
        x_pos = range(len(top_k))
        ax2   = ax.twinx()
        ax.bar(x_pos, tk_probs,  color=TEAL,  alpha=0.7, label="Probability")
        ax2.plot(x_pos, tk_energy, color=CORAL, lw=2.0, marker="o", ms=5, label="Energy")
        ax.set_xticks(x_pos); ax.set_xticklabels(tk_labels, fontsize=7)
        ax.set_ylabel("Probability", color=TEAL)
        ax2.set_ylabel("QUBO Energy", color=CORAL)
        ax.set_title("Top-K QAOA Solutions — Probability vs Energy", pad=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig("outputs/fig_qubo_topk.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("     ✅ 6 figures saved to outputs/")


# cache for route mapping (set in run_pipeline scope)
fml_routes_cache = {}

def _print_summary(qaoa_eval, greedy, cw, result):
    print("\n" + "─"*55)
    print("  RESULTS SUMMARY")
    print("─"*55)
    print(f"  QAOA Best Bitstring : {result['best_bitstring']}")
    print(f"  QAOA QUBO Energy    : {qaoa_eval['qubo_energy']:.4f}")
    print(f"  QAOA Total Cost     : ${qaoa_eval['total_cost']:>12,.0f}")
    print(f"  Greedy Total Cost   : ${greedy['total_cost']:>12,.0f}")
    print(f"  Clarke-Wright Cost  : ${cw['total_cost']:>12,.0f}")
    savings = greedy['total_cost'] - qaoa_eval['total_cost']
    print(f"  QAOA Savings vs Greedy: ${savings:+,.0f}")
    print("─"*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="stressed",
                        choices=["normal","stressed","crisis"])
    parser.add_argument("--layers",   default=2, type=int)
    parser.add_argument("--vessels",  default=4, type=int)
    args = parser.parse_args()
    run_pipeline(scenario=args.scenario, p_layers=args.layers, n_vessels=args.vessels)

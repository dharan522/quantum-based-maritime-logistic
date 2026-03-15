import os
import numpy as np
import matplotlib.pyplot as plt
import json


C = {
    "bg":    "#050E1A",
    "panel": "#071D2E",
    "teal":  "#00C9C8",
    "cyan":  "#00E5FF",
    "amber": "#FFB347",
    "coral": "#FF6B6B",
    "green": "#2ECC71",
    "text":  "#D6EAF8",
    "muted": "#8BAFC4",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor": C["bg"],  "axes.facecolor":  C["panel"],
        "axes.edgecolor":   C["teal"]+"44", "axes.labelcolor": C["text"],
        "xtick.color":  C["muted"],   "ytick.color":  C["muted"],
        "text.color":   C["text"],    "grid.color":   C["teal"]+"22",
        "grid.linestyle": "--",       "grid.alpha":   0.4,
        "font.family":  "monospace",  "axes.titlecolor": C["cyan"],
    })

def generate_all_charts(solver, result, qaoa_eval, top_k, greedy, cw, bench,
                         cong, berth_q, arrivals, fuel_px, Q, meta, g, b, E,
                         scenario, port_ids, twin, events, fml, tp):
    os.makedirs("outputs", exist_ok=True)
    _style()
    _fig_convergence(solver, g, b, E)
    _fig_digital_twin(cong, berth_q, port_ids, scenario)
    _fig_port_network(result, meta, fml)
    _fig_distribution_benchmark(result, bench)
    _fig_twin_operations(fuel_px, arrivals, tp)
    _fig_qubo_topk(Q, meta, top_k)


def _fig_convergence(solver, g, b, E):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(C["bg"])
    h = solver.energy_history
    ax[0].plot(range(len(h)), h, color=C["cyan"], lw=1.5)
    ax[0].fill_between(range(len(h)), h, alpha=0.15, color=C["teal"])
    ax[0].axhline(solver.optimal_energy, color=C["coral"], ls="--", lw=1.5,
                  label=f"optimal={solver.optimal_energy:.3f}")
    ax[0].set_title("QAOA Convergence")
    ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("⟨HC⟩")
    ax[0].legend(facecolor=C["panel"], fontsize=9); ax[0].grid(True)
    cf = ax[1].contourf(b, g, E, levels=22, cmap="plasma")
    fig.colorbar(cf, ax=ax[1]).set_label("⟨HC⟩", color=C["text"])
    ax[1].set_title("Energy Landscape (p=1)")
    ax[1].set_xlabel("β"); ax[1].set_ylabel("γ")
    plt.tight_layout()
    plt.savefig("outputs/fig_convergence_landscape.png", dpi=130, bbox_inches="tight")
    plt.close()


def _fig_digital_twin(cong, berth_q, port_ids, scenario):
    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    fig.patch.set_facecolor(C["bg"])
    sub = port_ids[:8]
    im  = ax[0].imshow(cong[sub].values.T, aspect="auto", cmap="hot", vmin=0, vmax=1)
    ax[0].set_yticks(range(len(sub))); ax[0].set_yticklabels(sub, fontsize=9)
    ax[0].set_xlabel("Hour"); ax[0].set_title(f"Port Congestion — {scenario.upper()}")
    fig.colorbar(im, ax=ax[0], fraction=0.02)
    ax[1].plot(berth_q.index, berth_q["utilisation"], color=C["teal"], lw=2, label="Utilisation")
    ax[1].fill_between(berth_q.index, berth_q["utilisation"], alpha=0.12, color=C["teal"])
    ax2 = ax[1].twinx()
    ax2.plot(berth_q.index, berth_q["wait_time_hr"], color=C["amber"], lw=1.5, ls="--", label="Wait hr")
    ax[1].set_xlabel("Hour"); ax[1].set_ylabel("Utilisation"); ax2.set_ylabel("Wait (hr)")
    ax[1].set_title("SHA Berth Queue (M/M/c)")
    ax[1].legend(facecolor=C["panel"], fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/fig_digital_twin.png", dpi=130, bbox_inches="tight")
    plt.close()


def _fig_port_network(result, meta, fml):
    try:
        with open("data/ports.json") as f:
            pj = json.load(f)
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(C["bg"]); ax.set_facecolor("#071520")
        ax.set_xlim(-130, 160); ax.set_ylim(-40, 65)
        ax.set_title("Port Network — QAOA Route Assignments")
        cols = [C["cyan"], C["amber"], C["coral"], C["green"]]
        for i in range(fml.n_v):
            for ri in range(fml.n_r):
                key = f"x_v{i}_r{ri}"
                if key in meta["var_labels"]:
                    qi = meta["var_labels"].index(key)
                    bs = result["best_bitstring"]
                    if qi < len(bs) and bs[qi] == "1" and ri < len(fml.routes[i]):
                        route = fml.routes[i][ri]["route"]
                        for k in range(len(route) - 1):
                            p1, p2 = route[k], route[k+1]
                            if p1 in pj and p2 in pj:
                                ax.annotate("", xy=(pj[p2]["lon"], pj[p2]["lat"]),
                                    xytext=(pj[p1]["lon"], pj[p1]["lat"]),
                                    arrowprops=dict(arrowstyle="->", color=cols[i%4], lw=1.8))
        for pid, d in pj.items():
            ax.scatter(d["lon"], d["lat"], s=80, color=C["cyan"], zorder=5,
                       edgecolors=C["bg"], lw=0.5)
            ax.annotate(pid, (d["lon"], d["lat"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=8, color=C["text"])
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.savefig("outputs/fig_port_network.png", dpi=130, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def _fig_distribution_benchmark(result, bench):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(C["bg"])
    counts = result["counts"]; total = sum(counts.values())
    top18  = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:18]
    lbls   = [c[0][::-1] for c in top18]
    probs  = [c[1]/total for c in top18]
    bar_c  = [C["cyan"] if l == result["best_bitstring"] else C["teal"]+"88" for l in lbls]
    ax[0].bar(range(len(lbls)), probs, color=bar_c, edgecolor=C["bg"])
    ax[0].set_xticks(range(len(lbls)))
    ax[0].set_xticklabels(lbls, rotation=45, ha="right", fontsize=6)
    ax[0].set_ylabel("Probability"); ax[0].set_title("Measurement Distribution")
    ax[0].grid(axis="y", alpha=0.3)
    methods = [r["method"] for r in bench.to_dict("records")]
    costs   = [r["total_cost"]/1e6 for r in bench.to_dict("records")]
    bc2     = [C["cyan"] if m=="QAOA" else C["amber"] if m=="Greedy-NN" else C["coral"] for m in methods]
    ax[1].bar(methods, costs, color=bc2, edgecolor=C["bg"], width=0.5)
    for i, (m, v) in enumerate(zip(methods, costs)):
        ax[1].text(i, v+0.02, f"${v:.2f}M", ha="center", fontsize=10, color=C["text"])
    ax[1].set_ylabel("Cost (USD M)"); ax[1].set_title("QAOA vs Classical")
    ax[1].set_ylim(0, max(costs)*1.3); ax[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/fig_distribution_benchmark.png", dpi=130, bbox_inches="tight")
    plt.close()


def _fig_twin_operations(fuel_px, arrivals, tp):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(C["bg"])
    ax[0].plot(range(len(fuel_px)), fuel_px, color=C["amber"], lw=2)
    ax[0].fill_between(range(len(fuel_px)), fuel_px, alpha=0.15, color=C["amber"])
    ax[0].axhline(650, color=C["muted"], ls="--", lw=1)
    ax[0].set_title("Bunker Fuel (GBM)"); ax[0].set_xlabel("Days")
    ax[0].set_ylabel("USD/tonne"); ax[0].grid(True)
    arr_c = arrivals.groupby("port").size().sort_values()
    ax[1].barh(arr_c.index, arr_c.values, color=C["teal"], alpha=0.85, edgecolor=C["bg"])
    ax[1].set_title("Vessel Arrivals"); ax[1].set_xlabel("Count")
    ax[1].grid(axis="x", alpha=0.3)
    stp = tp.sort_values("throughput_factor")
    bc3 = [C["coral"] if v<0.7 else C["amber"] if v<0.9 else C["green"]
           for v in stp["throughput_factor"]]
    ax[2].barh(stp.index, stp["throughput_factor"]*100, color=bc3, edgecolor=C["bg"])
    ax[2].set_title("Port Throughput %"); ax[2].set_xlabel("%")
    ax[2].set_xlim(0, 115); ax[2].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/fig_twin_operations.png", dpi=130, bbox_inches="tight")
    plt.close()


def _fig_qubo_topk(Q, meta, top_k):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C["bg"])
    lbl_s = [l.replace("x_","").replace("b_","b:") for l in meta["var_labels"]]
    im = ax[0].imshow(Q, cmap="RdBu", vmin=Q.min(), vmax=Q.max(), aspect="auto")
    ax[0].set_xticks(range(len(lbl_s)))
    ax[0].set_xticklabels(lbl_s, rotation=45, ha="right", fontsize=7)
    ax[0].set_yticks(range(len(lbl_s)))
    ax[0].set_yticklabels(lbl_s, fontsize=7)
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    ax[0].set_title(f"QUBO Matrix ({Q.shape[0]}×{Q.shape[0]})")
    if top_k:
        tk_p = [s["probability"] for s in top_k]
        tk_e = [s["energy"] for s in top_k]
        tk_l = [f"BS{i+1}" for i in range(len(top_k))]
        ax2  = ax[1].twinx()
        ax[1].bar(tk_l, tk_p, color=C["teal"], alpha=0.7)
        ax2.plot(tk_l, tk_e, color=C["coral"], lw=2, marker="o", ms=5)
        ax[1].set_ylabel("Probability", color=C["teal"])
        ax2.set_ylabel("Energy", color=C["coral"])
        ax[1].set_title("Top-K Solutions"); ax[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/fig_qubo_topk.png", dpi=130, bbox_inches="tight")
    plt.close()
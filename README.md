# ⚓ QAOA Maritime Logistics Optimiser + Digital Twin Port Simulation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.3.1-6929C4?logo=ibm)](https://qiskit.org)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A quantum-classical hybrid system that solves **Maritime Logistics Optimisation** — vessel route assignment, berth scheduling, and cargo matching — using the **Quantum Approximate Optimisation Algorithm (QAOA)** over a 22-variable QUBO, augmented by a **Digital Twin** that simulates real-world port operations (Poisson arrivals, M/M/c queuing, Markov weather, GBM fuel prices).

---

## 🧠 What This Solves

Traditional maritime logistics involves three tightly coupled NP-hard sub-problems:

| Sub-Problem | Classical Approach | This Project |
|---|---|---|
| Vessel Route Assignment | Greedy / Clarke-Wright | QAOA on QUBO |
| Berth Allocation | FCFS / Priority Queue | Digital Twin + Schedule |
| Cargo-Vessel Matching | Hungarian Algorithm | QUBO + Revenue objective |

All three are **fused into a single QUBO matrix** and solved jointly by QAOA.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   QAOA MARITIME LOGISTICS SYSTEM                         │
│                                                                          │
│  ┌─────────────────┐     ┌──────────────────────────────────────────┐    │
│  │  SYNTHETIC DATA │     │         DIGITAL TWIN LAYER               │    │
│  │  12 Global Ports│────▶│  ┌─────────────────────────────────────┐ │    │
│  │  8 Vessels      │     │  │ Poisson Vessel Arrivals             │ │    │
│  │  40 Cargo Items │     │  │ M/M/c Berth Queue Simulation       │ │    │
│  │  Distance Matrix│     │  │ Markov Weather State Machine       │ │    │
│  └─────────────────┘     │  │ GBM Bunker Fuel Price Simulation   │ │    │
│                           │  │ Stochastic Disruption Events       │ │    │
│                           │  └───────────────┬─────────────────────┘ │    │
│                           └──────────────────┼──────────────────────┘    │
│                                              │ Adjusted Costs             │
│                                              ▼                            │
│                           ┌─────────────────────────────┐                │
│                           │     QUBO FORMULATOR         │                │
│                           │  obj1: α · RouteCost(x)    │                │
│                           │  obj2: β · BerthPenalty(b) │                │
│                           │  obj3: -γ · Revenue(y)     │                │
│                           │  + Lagrange constraints     │                │
│                           └──────────────┬──────────────┘                │
│                                          │ Q matrix (22×22)              │
│              ┌───────────────────────────┴─────────────────────┐         │
│              │                                                  │         │
│              ▼                                                  ▼         │
│  ┌─────────────────────────────┐     ┌────────────────────────────────┐  │
│  │       QAOA SOLVER           │     │    CLASSICAL BENCHMARKS        │  │
│  │  |0⟩^n → H^n               │     │  Greedy Nearest-Neighbour      │  │
│  │  [HC(γ) + HB(β)] × p       │     │  Clarke-Wright Savings         │  │
│  │  COBYLA optimiser           │     │  Hungarian Cargo Matching      │  │
│  │  Statevector simulator      │     │  Priority Berth Schedule       │  │
│  └──────────────┬──────────────┘     └────────────────┬───────────────┘  │
│                 │ Best bitstring                       │                  │
│                 └──────────────────┬──────────────────┘                  │
│                                    ▼                                      │
│                  ┌─────────────────────────────────────┐                 │
│                  │        DASH DASHBOARD               │                 │
│                  │  Global Port Network Map            │                 │
│                  │  QAOA Convergence Curve             │                 │
│                  │  Port Congestion Heatmap            │                 │
│                  │  M/M/c Berth Queue Plot             │                 │
│                  │  Fuel Price GBM Simulation          │                 │
│                  │  Energy Landscape (γ-β)             │                 │
│                  │  QUBO Heatmap                       │                 │
│                  │  Probability Distribution           │                 │
│                  │  Cost Benchmark Chart               │                 │
│                  │  Disruption Event Log Table         │                 │
│                  │  Route Assignment Table             │                 │
│                  └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
qaoa_maritime/
├── main.py                      # Master pipeline (CLI)
├── dashboard.py                 # Interactive Dash dashboard
├── src/
│   ├── __init__.py
│   ├── qubo_formulator.py       # 3-objective QUBO (routing+berth+cargo)
│   ├── qaoa_solver.py           # QAOA circuit + COBYLA optimiser
│   ├── classical_solver.py      # Greedy, Clarke-Wright, Hungarian
│   └── digital_twin.py          # Port operations digital twin
├── data/
│   ├── generate_data.py         # Synthetic data generator
│   ├── ports.csv / ports.json   # 12 global port nodes
│   ├── vessels.csv              # 8-vessel fleet
│   ├── cargo.csv                # 40-item cargo manifest
│   ├── distance_matrix.csv      # Haversine distances (nm)
│   └── congestion.csv           # Baseline port congestion
├── outputs/
│   ├── fig_convergence_landscape.png
│   ├── fig_digital_twin.png
│   ├── fig_port_network.png
│   ├── fig_distribution_benchmark.png
│   ├── fig_twin_operations.png
│   ├── fig_qubo_topk.png
│   └── report.json
├── tests/
│   └── test_all.py              # 40 unit tests
├── .github/
│   └── workflows/ci.yml         # GitHub Actions (Python 3.10–3.12)
├── setup_and_run.py             # One-click auto setup
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Extract & Install
```bash
unzip qaoa_maritime.zip
cd qaoa_maritime
pip install -r requirements.txt
```

### 2. One-Click Setup (recommended)
```bash
python setup_and_run.py
```
This installs deps → generates data → runs tests → runs pipeline → shows results.

### 3. Manual Step-by-Step
```bash
# Generate data
python data/generate_data.py

# Run full pipeline
python main.py --scenario stressed --layers 2 --vessels 4

# Run tests
pytest tests/ -v

# Launch dashboard
python dashboard.py
# → http://127.0.0.1:8051
```

---

## ⚙ CLI Options

| Flag | Default | Options | Description |
|---|---|---|---|
| `--scenario` | `stressed` | `normal` / `stressed` / `crisis` | Digital Twin disruption level |
| `--layers` | `2` | `1–4` | QAOA circuit depth (p) |
| `--vessels` | `4` | `2–8` | Number of vessels to optimise |

---

## 🌐 Port Network (12 Nodes)

| ID | Port | Region | Berths |
|---|---|---|---|
| SHA | Shanghai | Asia | 8 |
| SIN | Singapore | Asia | 6 |
| HKG | Hong Kong | Asia | 5 |
| ROT | Rotterdam | Europe | 10 |
| HAM | Hamburg | Europe | 7 |
| ANT | Antwerp | Europe | 6 |
| LAX | Los Angeles | Americas | 9 |
| NYK | New York | Americas | 7 |
| DXB | Dubai (Jebel Ali) | Middle East | 8 |
| COL | Colombo | Asia | 5 |
| MOM | Mombasa | Africa | 4 |
| SAO | Santos | Americas | 6 |

---

## 🔬 QUBO Formulation

```
minimise:  α · Σᵥᵣ cost(v,r)·x[v,r]          ← routing cost
         + β · Σₚ congestion(p)·b[p]           ← berth penalty
         - γ · Σᵥᵣ revenue(v,r)·x[v,r]         ← cargo revenue
         + A · Σᵥ (Σᵣ x[v,r] - 1)²            ← one route per vessel
         + B · Σ_{p,q} b[p]·b[q]               ← berth capacity
subject to: x[v,r], b[p] ∈ {0,1}
```

Variables:
- `x_v{i}_r{j}` — vessel `i` assigned to route candidate `j`
- `b_{p1}_{p2}` — berth allocation flag for port-pair `(p1,p2)`

---

## 🔮 Digital Twin Simulation Layers

| Layer | Model | Output |
|---|---|---|
| Vessel Arrivals | Poisson process (λ per port) | Arrival time series |
| Berth Queue | M/M/c queueing theory | Utilisation + wait time |
| Weather | 5-state Markov chain | Clear/Cloudy/Rain/Storm/Fog |
| Fuel Prices | Geometric Brownian Motion | $/tonne time series |
| Disruptions | Stochastic event generator | Delay + cost multipliers |
| Port Tides | M2+S2 sinusoidal model | Tidal height series |

---

## 🌊 Disruption Scenarios

| Scenario | Canal Closure | Breakdown | Strike | Weather |
|---|---|---|---|---|
| normal | 1% | 3% | 2% | 10% |
| stressed | 3% | 8% | 6% | 20% |
| crisis | 20% | 15% | 15% | 40% |

---

## 🧪 Testing

```
40 passed in 5.61s

TestQUBOFormulator   —  9 tests  (shape, symmetry, labels, routes, evaluate, revenue)
TestQAOASolver       —  7 tests  (expectation, optimise, history, counts, landscape, top-k, probs)
TestClassicalSolver  —  7 tests  (greedy, CW, hungarian, berth, benchmark, no-overlap)
TestDigitalTwin      — 17 tests  (events, weather, arrivals, queue, tides, fuel, multipliers)
```

---

## 🏆 Resume Bullet Points

- Engineered a **multi-objective QUBO formulation** fusing 3 maritime sub-problems (vessel routing, berth allocation, cargo matching) into a 22-variable binary matrix and solved via QAOA on Qiskit Aer statevector simulator
- Built a **Maritime Port Digital Twin** implementing 6 simulation layers: Poisson vessel arrivals, M/M/c berth queuing, 5-state Markov weather, GBM fuel price, tidal models, and stochastic disruption events across 12 global ports
- Implemented **4 classical benchmark algorithms** (Greedy NN, Clarke-Wright Savings, Hungarian Matching, Priority Berth Scheduler) for rigorous QAOA performance comparison
- Developed an **11-panel interactive Dash dashboard** with a live global port map, QAOA convergence, energy landscape, QUBO heatmap, congestion heatmap, berth queue, and real-time scenario switching
- Delivered **40 unit tests** (100% pass rate) with full pytest coverage across QUBO, QAOA, classical solvers, and digital twin modules, with GitHub Actions CI across Python 3.10–3.12

---

## 📋 Requirements

```
Python ≥ 3.10
qiskit ≥ 2.0.0 · qiskit-aer ≥ 0.13.0 · scipy · numpy · pandas
matplotlib · plotly · dash · networkx
```

No paid APIs. Runs 100% locally.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

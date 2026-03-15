# ⚓ QAOA Maritime Logistics Optimiser

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.3.1-6929C4?logo=ibm)](https://qiskit.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A quantum-classical hybrid system that solves **Maritime Logistics Optimisation** — vessel route assignment, berth scheduling, and cargo matching — using the **Quantum Approximate Optimisation Algorithm (QAOA)** over a 22-variable QUBO, augmented by a **Digital Twin** that simulates real-world port operations including Poisson arrivals, M/M/c queuing, Markov weather chains, and GBM fuel price simulation.

---

## 🌐 Web Interface

The project runs as a full web application. Configure scenario, layers, and vessels in the sidebar, click **RUN QAOA**, and watch the pipeline execute live in the browser with streaming logs, progress bar, KPI cards, 6 charts, and result tables.

```bash
python app.py
```
Open → **http://127.0.0.1:5000**

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  QAOA MARITIME LOGISTICS SYSTEM                     │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │ Synthetic    │    │          Digital Twin Layer             │   │
│  │ Dataset      │───▶│  Poisson Arrivals · M/M/c Berth Queue  │   │
│  │ 12 ports     │    │  Markov Weather · GBM Fuel Prices      │   │
│  │  8 vessels   │    │  Stochastic Disruption Events          │   │
│  │ 40 cargoes   │    └──────────────────┬──────────────────────┘   │
│  └──────────────┘                       │ Adjusted costs            │
│                                         ▼                           │
│                        ┌───────────────────────────┐               │
│                        │      QUBO Formulator      │               │
│                        │  Route cost · Berth penalty│               │
│                        │  Revenue · Lagrange terms │               │
│                        └──────────────┬────────────┘               │
│                                       │ Q matrix (22×22)           │
│              ┌────────────────────────┴──────────────────┐         │
│              ▼                                            ▼         │
│   ┌─────────────────────┐              ┌───────────────────────┐   │
│   │    QAOA Solver      │              │  Classical Benchmarks │   │
│   │  Qiskit Aer · COBYLA│              │  Greedy · Clarke-Wright│  │
│   │  p-layer circuit    │              │  Hungarian · Berth Sch│   │
│   └──────────┬──────────┘              └───────────────────────┘   │
│              │ Best bitstring                                        │
│              ▼                                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    Flask Web App                             │  │
│   │  Live terminal log · Progress bar · KPI cards               │  │
│   │  6 charts · Route table · Event log · Top-K solutions       │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
qaoa_maritime/
├── app.py                      # Flask web server + pipeline runner
├── main.py                     # CLI pipeline (terminal mode)
├── dashboard.py                # Dash interactive dashboard
├── setup_and_run.py            # One-click install + run
├── requirements.txt            # All Python dependencies
│
├── templates/
│   └── index.html              # Web UI — HTML structure
│
├── static/
│   ├── style.css               # Web UI — all CSS styles
│   └── main.js                 # Web UI — all JavaScript
│
├── utils/
│   ├── __init__.py
│   └── charts.py               # All matplotlib chart generation
│
├── src/
│   ├── __init__.py
│   ├── qubo_formulator.py      # Markowitz → QUBO matrix (3 objectives)
│   ├── qaoa_solver.py          # QAOA circuit + COBYLA optimiser
│   ├── classical_solver.py     # Greedy, Clarke-Wright, Hungarian
│   └── digital_twin.py        # Port operations digital twin
│
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── ports.csv / ports.json  # 12 global port nodes
│   ├── vessels.csv             # 8-vessel fleet
│   ├── cargo.csv               # 40-item cargo manifest
│   ├── distance_matrix.csv     # Haversine distances (nautical miles)
│   └── congestion.csv          # Baseline port congestion
│
├── outputs/                    # Generated charts + report.json
│   ├── fig_convergence_landscape.png
│   ├── fig_digital_twin.png
│   ├── fig_port_network.png
│   ├── fig_distribution_benchmark.png
│   ├── fig_twin_operations.png
│   ├── fig_qubo_topk.png
│   └── report.json
│
├── tests/
│   └── test_all.py             # 40 unit tests
│
├── .github/
│   └── workflows/ci.yml        # GitHub Actions CI
│
├── LAB_LOGBOOK.md              # Day-by-day workflow + examiner Q&A
├── LICENSE                     # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/qaoa-maritime.git
cd qaoa-maritime
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Generate dataset
```bash
python data/generate_data.py
```

### 4. Run the web app
```bash
python app.py
```
Open → **http://127.0.0.1:5000**

---

## ⚙ CLI Mode (Terminal Only)

```bash
python main.py
```

With options:
```bash
python main.py --scenario crisis --layers 3 --vessels 6
```

| Flag | Default | Options | Description |
|---|---|---|---|
| `--scenario` | `stressed` | `normal` / `stressed` / `crisis` | Digital twin disruption level |
| `--layers` | `2` | `1 – 4` | QAOA circuit depth p |
| `--vessels` | `4` | `2 – 8` | Number of vessels to optimise |

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

```
40 passed in 5.6s

TestQUBOFormulator   —  9 tests
TestQAOASolver       —  7 tests
TestClassicalSolver  —  7 tests
TestDigitalTwin      — 17 tests
```

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
minimise:  α · Σ cost(v,r) · x[v,r]          ← routing cost
         + β · Σ congestion(p) · b[p]         ← berth penalty
         - γ · Σ revenue(v,r) · x[v,r]        ← cargo revenue
         + A · Σᵥ (Σᵣ x[v,r] - 1)²           ← one route per vessel
         + B · Σ b[p] · b[q]                  ← berth capacity

subject to:  x[v,r], b[p] ∈ {0,1}
```

Variables:
- `x_v{i}_r{j}` — vessel `i` assigned to route candidate `j`
- `b_{p1}_{p2}` — berth allocation flag for port-pair

---

## 🔮 Digital Twin Simulation Layers

| Layer | Model | Description |
|---|---|---|
| Vessel arrivals | Poisson(λ) | Per-port stochastic arrivals |
| Berth queue | M/M/c | Occupancy, wait time, utilisation |
| Weather | 5-state Markov chain | Clear / Cloudy / Rain / Storm / Fog |
| Fuel prices | Geometric Brownian Motion | Bunker cost per tonne |
| Disruptions | Stochastic event generator | 6 types: weather, congestion, breakdown, canal, strike, fog |
| Tides | M2 + S2 sinusoidal | Tidal height time series |

### Disruption Scenarios

| Scenario | Canal Closure | Breakdown | Strike | Weather |
|---|---|---|---|---|
| normal | 1% | 3% | 2% | 10% |
| stressed | 3% | 8% | 6% | 20% |
| crisis | 20% | 15% | 15% | 40% |

---

## 📊 Output Charts

| File | Description |
|---|---|
| `fig_convergence_landscape.png` | QAOA convergence + γ-β energy landscape |
| `fig_digital_twin.png` | Port congestion heatmap + M/M/c berth queue |
| `fig_port_network.png` | Global port map with QAOA route assignments |
| `fig_distribution_benchmark.png` | Measurement probability + cost benchmark |
| `fig_twin_operations.png` | Fuel prices + arrivals + port throughput |
| `fig_qubo_topk.png` | QUBO matrix heatmap + Top-K solutions |

---

## 🏆 Resume Bullet Points

- Engineered a **multi-objective QUBO formulation** fusing vessel routing, berth allocation, and cargo matching into a 22-variable binary matrix and solved via QAOA on Qiskit Aer statevector simulator
- Built a **Maritime Port Digital Twin** implementing 6 simulation layers: Poisson vessel arrivals, M/M/c berth queuing, 5-state Markov weather, GBM fuel price, tidal models, and stochastic disruption events across 12 global ports
- Developed a **Flask web application** with live streaming pipeline logs, real-time progress tracking, 6 matplotlib charts, and interactive result tables served at localhost
- Implemented **4 classical benchmark algorithms** (Greedy NN, Clarke-Wright Savings, Hungarian Matching, Priority Berth Scheduler) for rigorous QAOA comparison
- Delivered **40 unit tests** (100% pass rate) across QUBO, QAOA, classical solvers, and digital twin modules with GitHub Actions CI across Python 3.10–3.12

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Quantum | Qiskit 2.3, Qiskit Aer 0.17 |
| Optimisation | SciPy COBYLA, NumPy |
| Web server | Flask 3.x |
| Frontend | HTML5, CSS3, Vanilla JS |
| Charts | Matplotlib |
| Data | Pandas, NetworkX |
| Tests | Pytest |
| CI/CD | GitHub Actions |

---

## 📋 Requirements

```
Python >= 3.10
qiskit >= 2.0.0
qiskit-aer >= 0.13.0
flask >= 3.0.0
scipy >= 1.10.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
plotly >= 5.14.0
dash >= 2.10.0
networkx >= 3.0
pytest >= 7.0.0
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

# Lab Logbook — QAOA Maritime Logistics Optimiser + Digital Twin

## Project Title: Quantum-Classical Maritime Logistics Optimisation with Digital Twin Port Simulation
## Duration: 2–3 weeks | Stack: Python 3.12, Qiskit 2.3, Dash | Level: Advanced

---

## WEEK 1 — Theory, Problem Formulation, Data

### Day 1–2: Problem Decomposition & QUBO Theory

**Goal:** Decompose the maritime logistics problem into quantum-compatible form.

**Three sub-problems identified:**
1. **Vessel Route Assignment** — combinatorial, NP-hard, binary decision
2. **Berth Allocation** — scheduling, resource-constrained
3. **Cargo-Vessel Matching** — bipartite matching, revenue maximisation

**QUBO Fusion:**
All three are encoded into a single QUBO matrix `Q` where:
- Diagonal `Q[i,i]` = linear (single-variable) objective terms
- Off-diagonal `Q[i,j]` = quadratic (cross-variable) interaction terms

**Multi-objective QUBO:**
```
min:  α·Σcost(v,r)·x[v,r]   ← α=0.4, routing cost
    + β·Σcongestion(p)·b[p]  ← β=0.3, berth penalty
    - γ·Σrevenue(v,r)·x[v,r] ← γ=0.3, cargo revenue (negate = maximise)
    + A·Σᵥ(Σᵣ x[v,r]-1)²    ← A=10, one route per vessel
    + B·Σ b[p]·b[q]          ← B=8, berth capacity
```

**Variable types:**
- `x_v{i}_r{j}` ∈ {0,1}: vessel i on route j
- `b_{p1}_{p2}` ∈ {0,1}: berth allocation for port pair

**Output:** Handwritten derivation of expanded QUBO terms

---

### Day 3: Synthetic Data Generation

**File:** `data/generate_data.py`

**Steps:**
1. Define 12 global ports with lat/lon, berth count, handling rate
2. Define 8-vessel fleet (ULCV, VLCV, Feeder, Tanker types)
3. Compute Haversine distance matrix in nautical miles:
   `d = 2R·arcsin(√(sin²(Δφ/2) + cosφ₁·cosφ₂·sin²(Δλ/2)))`, R=3440 nm
4. Generate 40-item cargo manifest with priorities and revenue
5. Compute travel time matrices per vessel speed

**Verify:** `distance_matrix.csv` is symmetric, diagonal=0, all values > 0

---

### Day 4: QUBO Formulator

**File:** `src/qubo_formulator.py`

**Steps:**
1. Generate candidate routes: enumerate all 3-port combinations (o,w,d), rank by total cost
2. Build variable map: `x_v{i}_r{j}` for vessel-route, `b_{p1}_{p2}` for port-pair berths
3. Populate Q diagonal:
   - Route cost term: `Q[qi,qi] += α * cost_norm`
   - Berth term: `Q[qi,qi] += β * (1/(dist/1000 + 0.1))`
   - Revenue term: `Q[qi,qi] -= γ * (rev/total_rev)`
4. Add constraint terms:
   - Vessel assignment: `Q[qi,qi] += A*(1-2K)`, `Q[qi,qj] += 2A`
   - Berth capacity: `Q[qi,qj] += B*0.5`

**Verify:** Lower triangle of Q is all zeros; meta["n_vars"] matches len(var_labels)

---

### Day 5: Digital Twin Architecture

**File:** `src/digital_twin.py`

**Six simulation layers:**

| Layer | Method | Notes |
|-------|---------|-------|
| Vessel Arrivals | Poisson(λ) per port per day | λ from real throughput data |
| Berth Queue | M/M/c discrete event | λ=arrivals/hour, μ=1/6 berth/hr |
| Weather | 5-state Markov chain | Transition matrix calibrated to maritime data |
| Fuel | GBM: S(t+1)=S(t)·exp((μ-σ²/2)+σW) | σ=0.015, μ=0.0002 |
| Tides | M2+S2 sinusoidal: 2.5·sin(2πt/12.42)+0.8·sin(2πt/12) | Standard harmonic tidal model |
| Disruptions | Poisson events per scenario | 6 types, severity ∈ [0.2, 0.95] |

**Steps:**
1. Implement `generate_disruptions(scenario)` with 3 scenario modes
2. Implement M/M/c queue: track occupied_berths, queue_length, wait_time
3. Markov weather: transition state each hour via `rng.choice(p=TRANS[state])`
4. GBM fuel: `price[t] = price[t-1] * exp((μ-σ²/2) + σ·N(0,1))`
5. `compute_route_cost_multipliers()`: apply event cost_multiplier to affected port routes

**Verify:** Crisis scenario produces more events than normal; throughput ≤ 1.0 everywhere

---

## WEEK 2 — Quantum Circuit & Solvers

### Day 6–7: QAOA Circuit

**File:** `src/qaoa_solver.py`

**Circuit diagram (n=22 qubits, p=2 layers):**
```
|0⟩─H─[RZ(Q[i,i]·2γ)]─[CNOT·RZ(Q[i,j]·2γ)·CNOT]─[RX(2β)]─ ... ─M
```

**Cost Hamiltonian (HC):**
- Diagonal: `RZ(2·Q[i,i]·γ, i)` for all i
- Off-diagonal: `CNOT(i,j) → RZ(2·Q[i,j]·γ, j) → CNOT(i,j)` for i<j with Q[i,j]≠0

**Mixer Hamiltonian (HB):**
- Standard X-mixer: `RX(2β, i)` for all i
- Maintains ergodicity over binary solution space

**Expectation value:**
```
⟨HC⟩ = Σ_{x} P(x)·(xᵀQx)    computed via shot-based sampling
```

**COBYLA optimisation:**
- Gradient-free, handles noisy objective well
- `rhobeg=0.4` for initial trust region
- `maxiter=150` for convergence

**Key gotcha:** Qiskit returns bitstrings in reversed qubit order — always apply `[::-1]`

**Verify:** Energy should generally decrease over iterations; counts sum to shots×4

---

### Day 8: Classical Benchmarks

**File:** `src/classical_solver.py`

**Four algorithms:**

1. **Greedy Nearest-Neighbour**: for each vessel, enumerate all port pairs, pick minimum cost
2. **Clarke-Wright Savings**: `savings(i,j) = d(depot,i)+d(depot,j)-d(i,j)`, merge greedily
3. **Hungarian Algorithm**: `scipy.optimize.linear_sum_assignment` on revenue cost matrix
4. **Priority Berth Scheduler**: sort cargo by CRITICAL>HIGH>MEDIUM>LOW, earliest-free-berth

**Verify:**
- Greedy assigns all vessels, total_cost > 0
- Hungarian does not exceed vessel capacity
- Berth schedule has no overlapping slots on same berth

---

### Day 9: Pipeline Integration

**File:** `main.py`

**7-step pipeline:**
1. Digital twin → generate disruptions, congestion, arrivals, fuel prices, berth queue
2. QUBO formulator → apply adjusted costs from digital twin, build Q matrix
3. QAOA solver → optimise, get best bitstring + convergence history
4. Classical benchmarks → greedy, Clarke-Wright, Hungarian, benchmark table
5. Energy landscape → sub-QUBO scan for γ-β visualisation
6. Plot generation → 6 figures saved to outputs/
7. JSON report → full results serialised

**Test command:**
```bash
python main.py --scenario crisis --layers 3 --vessels 4
```

---

## WEEK 3 — Dashboard, Tests, Documentation

### Day 10–11: Dash Dashboard

**File:** `dashboard.py`

**11-panel layout:**
```
[KPI Bar: Energy | Cost | Vessels | Constraints | Layers | Scenario]
[Global Port Map (Scattergeo)]    [Convergence Curve]
[Congestion Heatmap — full width]
[M/M/c Berth Queue]               [GBM Fuel Price]
[Energy Landscape]                 [QUBO Heatmap]
[Probability Distribution]         [Cost Benchmark]
[Disruption Event Log Table]
[Route Assignment Table]
```

**Design system:** Deep ocean dark theme
- Background: #050E1A, Panel: #071D2E, Accent: #00C9C8 / #00E5FF
- Font: Courier New (monospace, technical aesthetic)
- Borders: #143252 (subtle navy)

**Key implementation:**
- `dcc.Store` passes full results dict between callbacks
- Geo map uses `go.Scattergeo` with natural earth projection
- All charts share `PLOTLY_BASE` layout dict for theme consistency
- Route table uses `style_data_conditional` to highlight selected routes in teal

---

### Day 12: Unit Tests

**File:** `tests/test_all.py`

**40 tests across 4 classes:**

```
TestQUBOFormulator  (9)  — shape, triangular, meta, labels, routes, evaluate, revenue
TestQAOASolver      (7)  — float, bitstring, history, counts, landscape, top-k, probs
TestClassicalSolver (7)  — greedy, CW, hungarian, berth columns, benchmark, no-overlap
TestDigitalTwin    (17)  — events, crisis>normal, weather shape, states, arrivals,
                           berth shape, utilisation range, tides, fuel>0, multipliers,
                           throughput<1, delay>0, congestion, all ports, range, route cost, summary
```

**Run:** `pytest tests/ -v` → **40 passed in 5.6s**

---

### Day 13: GitHub Prep

```bash
git init
git add .
git commit -m "feat: QAOA maritime logistics optimiser with digital twin"
git remote add origin https://github.com/yourusername/qaoa-maritime.git
git push -u origin main
```

CI automatically triggers on push:
- Python 3.10, 3.11, 3.12 matrix
- Data generation → pytest → pipeline smoke test → flake8

---

### Day 14: Demo Walkthrough

**For viva / demo presentation:**

```bash
# Step 1: Show data
python data/generate_data.py
# → Explain 12 ports, 8 vessels, 40 cargo, distance matrix

# Step 2: Normal run (fast)
python main.py --scenario normal --layers 1 --vessels 3
# → Show convergence, route assignments, comparison

# Step 3: Full stressed run
python main.py --scenario stressed --layers 2 --vessels 4
# → Show digital twin impact, QAOA vs classical

# Step 4: Crisis scenario
python main.py --scenario crisis --layers 2 --vessels 4
# → Dramatic throughput reduction, higher costs

# Step 5: Live dashboard
python dashboard.py
# → Walk through each panel interactively
```

---

## Architecture Decision Log

| Decision | Option A | Option B | Chosen | Reason |
|---|---|---|---|---|
| Optimiser | COBYLA | SPSA | **COBYLA** | Gradient-free, no shot overhead for gradient |
| Mixer | X-mixer | XY-mixer | **X-mixer** | Standard, sufficient for binary problems |
| Constraint encoding | Lagrange penalty | Cirq's mixer | **Lagrange** | Hardware-agnostic, clean QUBO |
| Berth model | FCFS queue | M/M/c | **M/M/c** | Analytically grounded, closed-form ρ |
| Fuel model | Random walk | GBM | **GBM** | Multiplicative, log-normal, standard in finance |
| Weather model | Random | Markov chain | **Markov chain** | Captures persistence (storms last) |

---

## Examiner Q&A

**Q: What makes this problem quantum-suitable?**
A: The vessel route assignment is a binary combinatorial problem with 2^n solution space. QAOA explores this space in quantum superposition, applying the cost Hamiltonian as a phase rotation and the mixer as a transverse field. For n=22 variables, the classical exhaustive search space is 2^22 = 4M states. QAOA provides a polynomial-depth variational approach.

**Q: How does the Digital Twin affect the quantum circuit?**
A: The Digital Twin adjusts the cost coefficients in the QUBO matrix before circuit construction. Specifically, `total_cost` per route is multiplied by the `cost_multiplier` from active disruption events. This means the Hamiltonian HC itself changes — the quantum circuit is solving a dynamically updated problem reflecting real-world port conditions.

**Q: Why fuse all three sub-problems into one QUBO?**
A: Solving them separately ignores cross-dependencies. For example, assigning a vessel to a congested port reduces berth throughput, which affects cargo loading schedule, which affects revenue. The fused QUBO captures these interactions via the off-diagonal terms Q[i,j], allowing QAOA to find joint solutions that optimise all three objectives simultaneously.

**Q: What is the M/M/c model and why is it appropriate?**
A: M/M/c is a classic queueing model: M (Markovian/Poisson) arrivals, M (exponential) service times, c parallel servers (berths). Port vessel arrivals follow a Poisson process well-documented in maritime literature. Exponential service times are a reasonable approximation for berth operations. The model gives analytically tractable utilisation ρ = λ/(cμ) and average queue length.

**Q: What is the QAOA energy landscape telling us?**
A: The γ-β landscape shows the expectation value ⟨HC⟩ as a function of the variational angles. Deep basins correspond to good parameter regions. For p=1, the landscape is smooth enough for COBYLA to find a minimum reliably. As p increases, the landscape becomes more complex with sharper minima, potentially yielding better solutions but requiring more iterations.

**Q: How does this scale to real quantum hardware?**
A: The 22-qubit circuit is within reach of current NISQ devices (IBM has 127+ qubit processors). On real hardware, gate noise and decoherence would require: (1) Zero-Noise Extrapolation (ZNE) or Probabilistic Error Cancellation (PEC) for error mitigation, (2) transpilation to native gate set (ECR/CZ), (3) possibly fewer layers (p=1 or 2) to stay within coherence time. The QUBO formulation itself is hardware-agnostic.

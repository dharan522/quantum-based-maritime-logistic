"""
qaoa_solver.py
QAOA circuit builder and optimiser for Maritime Logistics QUBO.

Circuit structure:
  |0⟩^n  →  H^n  →  [HC(γ)]^p  →  [HB(β)]^p  →  Measure

Cost Hamiltonian HC: encodes QUBO via RZ (diagonal) + CNOT-RZ-CNOT (off-diagonal)
Mixer Hamiltonian HB: standard X-mixer via RX gates
Optimiser: COBYLA (gradient-free, robust on noisy landscapes)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


class MaritimeQAOASolver:

    def __init__(
        self,
        qubo:       np.ndarray,
        var_labels: List[str],
        p_layers:   int   = 2,
        shots:      int   = 2048,
        optimizer:  str   = "COBYLA",
        seed:       int   = 2026,
    ):
        self.Q          = qubo
        self.labels     = var_labels
        self.n          = qubo.shape[0]
        self.p          = p_layers
        self.shots      = shots
        self.opt_name   = optimizer
        self.seed       = seed
        self.backend    = AerSimulator(method="statevector")

        self.energy_history: List[float]        = []
        self.param_history:  List[np.ndarray]   = []
        self.iteration      = 0
        self.optimal_params: Optional[np.ndarray] = None
        self.optimal_energy: Optional[float]      = None
        self.counts:         Optional[Dict]        = None

    # ── Circuit ──────────────────────────────────────────────────────────

    def _build_circuit(self, gamma: List[float], beta: List[float]) -> QuantumCircuit:
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))

        for layer in range(self.p):
            g = gamma[layer]
            # Diagonal ZZ terms
            for i in range(self.n):
                if abs(self.Q[i, i]) > 1e-10:
                    qc.rz(2 * self.Q[i, i] * g, i)
            # Off-diagonal ZZ interactions
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if abs(self.Q[i, j]) > 1e-10:
                        qc.cx(i, j)
                        qc.rz(2 * self.Q[i, j] * g, j)
                        qc.cx(i, j)
            # X-mixer
            b = beta[layer]
            for i in range(self.n):
                qc.rx(2 * b, i)

        qc.measure_all()
        return qc

    # ── Expectation ──────────────────────────────────────────────────────

    def _expectation(self, params: np.ndarray) -> float:
        gamma = params[:self.p]
        beta  = params[self.p:]
        qc    = self._build_circuit(gamma, beta)

        job    = self.backend.run(qc, shots=self.shots, seed_simulator=self.seed)
        counts = job.result().get_counts()

        energy = 0.0
        total  = sum(counts.values())
        for bs, cnt in counts.items():
            x = np.array([int(b) for b in bs[::-1]])
            energy += (cnt / total) * float(x @ self.Q @ x)

        self.energy_history.append(energy)
        self.param_history.append(params.copy())
        self.iteration += 1
        return energy

    # ── Optimise ─────────────────────────────────────────────────────────

    def optimise(
        self,
        init_gamma: Optional[List[float]] = None,
        init_beta:  Optional[List[float]] = None,
        maxiter:    int = 200,
    ) -> Dict:
        rng = np.random.default_rng(self.seed)
        if init_gamma is None:
            init_gamma = rng.uniform(0, np.pi, self.p).tolist()
        if init_beta is None:
            init_beta  = rng.uniform(0, np.pi / 2, self.p).tolist()

        x0 = np.array(init_gamma + init_beta, dtype=float)

        res = minimize(
            self._expectation, x0,
            method=self.opt_name,
            options={"maxiter": maxiter, "rhobeg": 0.4},
        )

        self.optimal_params = res.x
        self.optimal_energy = res.fun

        # Final high-shot sampling
        gamma_opt = self.optimal_params[:self.p]
        beta_opt  = self.optimal_params[self.p:]
        qc        = self._build_circuit(gamma_opt, beta_opt)
        job       = self.backend.run(qc, shots=self.shots * 4, seed_simulator=self.seed)
        self.counts = job.result().get_counts()

        best_bs_raw = max(self.counts, key=self.counts.get)
        best_bs     = best_bs_raw[::-1]

        return {
            "optimal_gamma":  gamma_opt.tolist(),
            "optimal_beta":   beta_opt.tolist(),
            "optimal_energy": float(self.optimal_energy),
            "best_bitstring": best_bs,
            "n_iterations":   self.iteration,
            "n_layers":       self.p,
            "converged":      res.success,
            "counts":         self.counts,
        }

    # ── Energy Landscape Scan ────────────────────────────────────────────

    def scan_landscape(self, resolution: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gammas = np.linspace(0, 2 * np.pi, resolution)
        betas  = np.linspace(0, np.pi,     resolution)
        E      = np.zeros((resolution, resolution))
        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                params  = np.array([g] * self.p + [b] * self.p)
                E[i, j] = self._expectation(params)
        return gammas, betas, E

    # ── Top-K Bitstrings ─────────────────────────────────────────────────

    def top_k_solutions(self, k: int = 5) -> List[Dict]:
        if self.counts is None:
            return []
        total   = sum(self.counts.values())
        sorted_ = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        results = []
        for bs_raw, cnt in sorted_[:k]:
            bs = bs_raw[::-1]
            x  = np.array([int(b) for b in bs])
            e  = float(x @ self.Q @ x)
            results.append({
                "bitstring":   bs,
                "probability": round(cnt / total, 4),
                "energy":      round(e, 4),
                "count":       cnt,
            })
        return results

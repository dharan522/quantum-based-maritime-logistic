"""
qubo_formulator.py
Converts Maritime Logistics Optimisation into a QUBO matrix for QAOA.

THREE sub-problems fused into one QUBO:
─────────────────────────────────────────────────────────────────────
1. VESSEL ROUTE ASSIGNMENT
   Binary variable x[v,r] = 1 if vessel v is assigned route r
   Objective: minimise total voyage cost (distance × fuel × cost/day)
   Constraint: each vessel assigned exactly one route

2. PORT BERTH ALLOCATION
   Binary variable b[p,t] = 1 if berth at port p is occupied at slot t
   Objective: minimise berth waiting time + maximise throughput
   Constraint: berths not over-committed

3. CARGO-VESSEL MATCHING
   Binary variable y[c,v] = 1 if cargo c is loaded on vessel v
   Objective: maximise revenue-weighted cargo delivery
   Constraint: vessel capacity not exceeded

Combined QUBO:
  min  α·Cost(x) + β·BerthPenalty(b) - γ·Revenue(y)
       + A·VesselConstraint(x) + B·BerthConstraint(b) + C·CapacityConstraint(y)
─────────────────────────────────────────────────────────────────────
For QAOA tractability, we use a REDUCED formulation:
  n_vessels × n_route_candidates binary variables  (vessel routing)
  + port-pair berth allocation variables            (port scheduling)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Tuple, Dict, List


class MaritimeQUBOFormulator:

    def __init__(
        self,
        distance_matrix: pd.DataFrame,
        vessels:         pd.DataFrame,
        cargo:           pd.DataFrame,
        ports:           pd.DataFrame,
        n_vessels:       int   = 4,
        n_routes:        int   = 3,
        alpha:           float = 0.4,   # cost weight
        beta:            float = 0.3,   # berth weight
        gamma:           float = 0.3,   # revenue weight
        penalty_A:       float = 10.0,  # vessel assignment penalty
        penalty_B:       float = 8.0,   # berth capacity penalty
        penalty_C:       float = 12.0,  # cargo capacity penalty
    ):
        self.D        = distance_matrix
        self.vessels  = vessels.head(n_vessels).reset_index(drop=True)
        self.cargo    = cargo
        self.ports    = ports
        self.port_ids = list(distance_matrix.index)
        self.n_v      = n_vessels
        self.n_r      = n_routes
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self.A        = penalty_A
        self.B        = penalty_B
        self.C        = penalty_C

        # Generate candidate routes (top-3 shortest paths for each vessel)
        self.routes      = self._generate_routes()
        self.var_map     = self._build_variable_map()
        self.n_vars      = len(self.var_map)
        self.var_labels  = list(self.var_map.keys())

    # ── Route Generation ────────────────────────────────────────────────

    def _generate_routes(self) -> Dict:
        """
        For each vessel, generate n_routes candidate routes as
        (origin, waypoint, destination) triples ranked by total distance.
        """
        routes = {}
        port_list = self.port_ids[:8]  # use top-8 ports for tractability

        for vi, vessel in self.vessels.iterrows():
            vessel_routes = []
            # Build all 3-port routes and rank by distance
            for o, w, d in combinations(port_list, 3):
                dist = (self.D.loc[o, w] + self.D.loc[w, d])
                spd  = vessel["speed_knots"]
                days = dist / spd / 24
                fuel_cost = days * vessel["fuel_tday"] * 650  # $650/tonne bunker
                op_cost   = days * vessel["cost_day"]
                total     = fuel_cost + op_cost
                vessel_routes.append({
                    "route":      (o, w, d),
                    "distance_nm": dist,
                    "days":       round(days, 2),
                    "total_cost": round(total, 0),
                })
            # Keep top-n_routes shortest routes
            vessel_routes.sort(key=lambda x: x["total_cost"])
            routes[vi] = vessel_routes[:self.n_r]

        return routes

    # ── Variable Map ────────────────────────────────────────────────────

    def _build_variable_map(self) -> Dict:
        """
        Build ordered map: variable_label → qubo_index
        Variables:
          x_v{i}_r{j}  : vessel i assigned to route j
          b_{p1}_{p2}  : berth allocation flag for port-pair (top-6)
        """
        var_map = {}
        idx     = 0

        # Vessel-route variables
        for vi in range(self.n_v):
            for ri in range(self.n_r):
                var_map[f"x_v{vi}_r{ri}"] = idx
                idx += 1

        # Berth variables (top-6 busiest port-pairs)
        top_ports = self.port_ids[:6]
        for p1, p2 in combinations(top_ports, 2):
            dist = self.D.loc[p1, p2]
            if dist < 5000:  # regional pairs only
                var_map[f"b_{p1}_{p2}"] = idx
                idx += 1

        return var_map

    # ── QUBO Construction ────────────────────────────────────────────────

    def build_qubo(self) -> Tuple[np.ndarray, Dict]:
        n = self.n_vars
        Q = np.zeros((n, n))

        # ── OBJECTIVE 1: Minimise vessel routing cost ──────────────────
        for vi in range(self.n_v):
            for ri in range(self.n_r):
                key  = f"x_v{vi}_r{ri}"
                cost = self.routes[vi][ri]["total_cost"]
                # Normalise to [0,1]
                cost_norm = cost / 1e6
                qi = self.var_map[key]
                Q[qi, qi] += self.alpha * cost_norm

        # ── OBJECTIVE 2: Berth allocation — penalise congested pairs ───
        for key, qi in self.var_map.items():
            if key.startswith("b_"):
                parts = key.split("_")
                p1, p2 = parts[1], parts[2]
                dist   = self.D.loc[p1, p2]
                # Closer ports = more congestion interaction
                congest_penalty = self.beta * (1.0 / (dist / 1000 + 0.1))
                Q[qi, qi] += congest_penalty

        # ── OBJECTIVE 3: Revenue — reward high-value routes ────────────
        total_revenue = self.cargo["revenue_usd"].sum()
        for vi in range(self.n_v):
            cap = self.vessels.loc[vi, "capacity_teu"]
            for ri in range(self.n_r):
                route = self.routes[vi][ri]["route"]
                # Cargo matching: revenue from cargo between route ports
                rev = self._route_revenue(route, cap)
                key = f"x_v{vi}_r{ri}"
                qi  = self.var_map[key]
                Q[qi, qi] -= self.gamma * (rev / total_revenue)

        # ── CONSTRAINT: Each vessel assigned exactly one route ─────────
        # A * (Σⱼ x_v{i}_r{j} - 1)²  for each vessel i
        for vi in range(self.n_v):
            route_indices = [self.var_map[f"x_v{vi}_r{ri}"] for ri in range(self.n_r)]
            for qi in route_indices:
                Q[qi, qi] += self.A * (1 - 2)   # -1 from expanding (Σ-1)²
            for qi, qj in combinations(route_indices, 2):
                i_, j_ = min(qi,qj), max(qi,qj)
                Q[i_, j_] += self.A * 2

        # ── CONSTRAINT: Berth pairs not simultaneously overloaded ──────
        berth_indices = [qi for key, qi in self.var_map.items() if key.startswith("b_")]
        if len(berth_indices) >= 2:
            for qi, qj in combinations(berth_indices[:6], 2):
                i_, j_ = min(qi,qj), max(qi,qj)
                Q[i_, j_] += self.B * 0.5

        meta = {
            "n_vars":         n,
            "n_vessels":      self.n_v,
            "n_routes":       self.n_r,
            "qubo_shape":     Q.shape,
            "qubo_range":     (float(Q.min()), float(Q.max())),
            "var_labels":     self.var_labels,
        }
        return Q, meta

    # ── Revenue Estimation ───────────────────────────────────────────────

    def _route_revenue(self, route: tuple, capacity_teu: int) -> float:
        """Estimate revenue from cargo that can be served on this route."""
        o, w, d = route
        route_ports = {o, w, d}
        matching = self.cargo[
            self.cargo["origin"].isin(route_ports) &
            self.cargo["destination"].isin(route_ports)
        ]
        loaded_teu = 0
        revenue    = 0.0
        for _, row in matching.iterrows():
            if loaded_teu + row["weight_teu"] <= capacity_teu:
                loaded_teu += row["weight_teu"]
                revenue    += row["revenue_usd"]
        return revenue

    # ── Solution Evaluation ─────────────────────────────────────────────

    def evaluate_solution(self, bitstring: str) -> Dict:
        x    = np.array([int(b) for b in bitstring])
        Q, _ = self.build_qubo()
        energy = float(x @ Q @ x)

        # Decode vessel assignments
        assignments = {}
        for vi in range(self.n_v):
            vessel_name = self.vessels.loc[vi, "name"]
            assigned    = None
            for ri in range(self.n_r):
                key = f"x_v{vi}_r{ri}"
                qi  = self.var_map[key]
                if qi < len(x) and x[qi] == 1:
                    assigned = self.routes[vi][ri]
                    break
            assignments[vessel_name] = assigned

        # Total cost and revenue
        total_cost = sum(
            v["total_cost"] for v in assignments.values() if v is not None
        )
        n_assigned = sum(1 for v in assignments.values() if v is not None)

        # Constraint check: each vessel has exactly one route
        valid = all(
            sum(x[self.var_map[f"x_v{vi}_r{ri}"]] for ri in range(self.n_r)) == 1
            for vi in range(self.n_v)
            if all(f"x_v{vi}_r{ri}" in self.var_map for ri in range(self.n_r))
        )

        return {
            "bitstring":    bitstring,
            "qubo_energy":  round(energy, 4),
            "assignments":  assignments,
            "n_assigned":   n_assigned,
            "total_cost":   round(total_cost, 0),
            "constraints_satisfied": valid,
        }

    def get_route_summary(self) -> pd.DataFrame:
        rows = []
        for vi in range(self.n_v):
            vessel = self.vessels.loc[vi, "name"]
            for ri, r in enumerate(self.routes[vi]):
                rows.append({
                    "vessel":      vessel,
                    "route_idx":   ri,
                    "route":       " → ".join(r["route"]),
                    "distance_nm": r["distance_nm"],
                    "days":        r["days"],
                    "cost_usd":    r["total_cost"],
                    "var":         f"x_v{vi}_r{ri}",
                })
        return pd.DataFrame(rows)

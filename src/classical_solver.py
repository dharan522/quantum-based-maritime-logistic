"""
classical_solver.py
Classical solvers for maritime logistics benchmarking against QAOA.

Implements:
1. Greedy Nearest-Neighbour route assignment
2. Savings Algorithm (Clarke-Wright) for vehicle routing
3. Hungarian Algorithm for vessel-cargo matching
4. Priority-based berth scheduling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


class ClassicalMaritimeSolver:

    def __init__(
        self,
        distance_matrix: pd.DataFrame,
        vessels:         pd.DataFrame,
        cargo:           pd.DataFrame,
        ports:           pd.DataFrame,
    ):
        self.D       = distance_matrix
        self.vessels = vessels
        self.cargo   = cargo
        self.ports   = ports
        self.port_ids= list(distance_matrix.index)

    # ── 1. Greedy Nearest-Neighbour ─────────────────────────────────────

    def greedy_route_assignment(self, n_vessels: int = 4) -> Dict:
        """
        Assign each vessel to its cheapest route greedily.
        Route = (origin, intermediate, destination) minimising total distance.
        """
        vessels  = self.vessels.head(n_vessels)
        port_sub = self.port_ids[:8]
        results  = {}
        total_cost = 0.0

        for _, v in vessels.iterrows():
            best_cost  = np.inf
            best_route = None
            spd = v["speed_knots"]

            for o in port_sub:
                for d in port_sub:
                    if o == d:
                        continue
                    dist  = self.D.loc[o, d]
                    days  = dist / spd / 24
                    cost  = days * (v["fuel_tday"] * 650 + v["cost_day"])
                    if cost < best_cost:
                        best_cost  = cost
                        best_route = (o, d)

            results[v["name"]] = {
                "route":    best_route,
                "cost_usd": round(best_cost, 0),
                "method":   "Greedy-NN",
            }
            total_cost += best_cost

        return {"assignments": results, "total_cost": round(total_cost, 0), "method": "Greedy-NN"}

    # ── 2. Clarke-Wright Savings ─────────────────────────────────────────

    def clarke_wright_savings(self, depot: str = "SHA", n_vessels: int = 4) -> Dict:
        """
        Clarke-Wright savings algorithm for multi-port routing.
        Savings(i,j) = d(depot,i) + d(depot,j) - d(i,j)
        """
        port_sub  = [p for p in self.port_ids[:8] if p != depot]
        # Compute savings
        savings   = {}
        for i in port_sub:
            for j in port_sub:
                if i < j:
                    s = (self.D.loc[depot, i] + self.D.loc[depot, j]
                         - self.D.loc[i, j])
                    savings[(i, j)] = s

        # Sort descending
        sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)

        # Build routes by merging highest-savings pairs
        routes  = [[p] for p in port_sub[:n_vessels]]
        used    = set()
        for (i, j), s in sorted_savings:
            if len(routes) == 0:
                break
            merged = False
            for r in routes:
                if r[-1] == i and j not in used:
                    r.append(j)
                    used.add(j)
                    merged = True
                    break
                elif r[-1] == j and i not in used:
                    r.append(i)
                    used.add(i)
                    merged = True
                    break
            if not merged and len(routes) < n_vessels:
                if i not in used and j not in used:
                    routes.append([i, j])
                    used.add(i)
                    used.add(j)

        # Build full routes through depot
        full_routes  = [[depot] + r + [depot] for r in routes[:n_vessels]]
        vessels_sub  = self.vessels.head(n_vessels)
        assignments  = {}
        total_cost   = 0.0

        for idx, (_, v) in enumerate(vessels_sub.iterrows()):
            r    = full_routes[idx] if idx < len(full_routes) else [depot, depot]
            dist = sum(self.D.loc[r[k], r[k+1]] for k in range(len(r)-1))
            days = dist / v["speed_knots"] / 24
            cost = days * (v["fuel_tday"] * 650 + v["cost_day"])
            assignments[v["name"]] = {
                "route":     " → ".join(r),
                "dist_nm":   round(dist, 0),
                "days":      round(days, 2),
                "cost_usd":  round(cost, 0),
                "method":    "Clarke-Wright",
            }
            total_cost += cost

        return {"assignments": assignments, "total_cost": round(total_cost, 0), "method": "Clarke-Wright"}

    # ── 3. Hungarian Vessel-Cargo Matching ──────────────────────────────

    def hungarian_cargo_assignment(self, n_vessels: int = 4) -> Dict:
        """
        Use Hungarian algorithm to match vessels to cargo batches
        maximising total revenue subject to capacity.
        """
        vessels_sub = self.vessels.head(n_vessels)
        n_v  = len(vessels_sub)
        n_c  = min(len(self.cargo), n_v * 3)
        cargo_sub = self.cargo.head(n_c)

        # Cost matrix (negative revenue for maximisation)
        cost_mat = np.zeros((n_v, n_c))
        for vi, (_, v) in enumerate(vessels_sub.iterrows()):
            for ci, (_, c) in enumerate(cargo_sub.iterrows()):
                if c["weight_teu"] <= v["capacity_teu"]:
                    cost_mat[vi, ci] = -c["revenue_usd"]
                else:
                    cost_mat[vi, ci] = 1e9   # infeasible

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        assignments = {}
        total_rev   = 0.0

        for vi, ci in zip(row_ind, col_ind):
            if cost_mat[vi, ci] < 1e8:
                v_name = vessels_sub.iloc[vi]["name"]
                c_row  = cargo_sub.iloc[ci]
                rev    = c_row["revenue_usd"]
                assignments[v_name] = {
                    "cargo_id":   c_row["cargo_id"],
                    "origin":     c_row["origin"],
                    "destination":c_row["destination"],
                    "weight_teu": c_row["weight_teu"],
                    "revenue":    round(rev, 2),
                }
                total_rev += rev

        return {"assignments": assignments, "total_revenue": round(total_rev, 2), "method": "Hungarian"}

    # ── 4. Priority Berth Scheduler ──────────────────────────────────────

    def priority_berth_schedule(self, port_id: str, n_berths: int, n_slots: int = 12) -> pd.DataFrame:
        """
        Schedule vessel berth allocations using priority queue.
        CRITICAL > HIGH > MEDIUM > LOW.
        """
        prio_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        port_cargo = self.cargo[
            (self.cargo["origin"] == port_id) | (self.cargo["destination"] == port_id)
        ].copy()
        port_cargo["prio_num"] = port_cargo["priority"].map(prio_order)
        port_cargo = port_cargo.sort_values("prio_num")

        schedule   = []
        berth_free = [0] * n_berths   # time slot when each berth becomes free

        for _, c in port_cargo.iterrows():
            # Find earliest free berth
            earliest = min(range(n_berths), key=lambda b: berth_free[b])
            start    = berth_free[earliest]
            duration = max(1, c["weight_teu"] // 500)   # 1 slot per 500 TEU
            end      = start + duration

            if start < n_slots:
                schedule.append({
                    "cargo_id":  c["cargo_id"],
                    "priority":  c["priority"],
                    "berth":     earliest + 1,
                    "start":     start,
                    "end":       min(end, n_slots),
                    "duration":  duration,
                    "weight_teu":c["weight_teu"],
                })
                berth_free[earliest] = end

        return pd.DataFrame(schedule)

    # ── Benchmark Summary ────────────────────────────────────────────────

    def benchmark_summary(self, qaoa_cost: float, n_vessels: int = 4) -> pd.DataFrame:
        greedy = self.greedy_route_assignment(n_vessels)
        cw     = self.clarke_wright_savings(n_vessels=n_vessels)
        return pd.DataFrame([
            {"method": "QAOA",          "total_cost": qaoa_cost,           "type": "quantum"},
            {"method": "Greedy-NN",     "total_cost": greedy["total_cost"],"type": "classical"},
            {"method": "Clarke-Wright", "total_cost": cw["total_cost"],    "type": "classical"},
        ])

"""
digital_twin.py
Maritime Port Digital Twin Simulation.

Models the physical port ecosystem as a virtual counterpart:
  ┌──────────────────────────────────────────────────────────────────┐
  │  PHYSICAL LAYER         DIGITAL TWIN LAYER                      │
  │  Vessel arrivals   →    Stochastic arrival process (Poisson)    │
  │  Berth occupancy   →    M/M/c queuing model simulation          │
  │  Crane operations  →    Monte Carlo throughput simulation       │
  │  Weather events    →    Markov chain weather state machine      │
  │  Channel tides     →    Sinusoidal tidal model                  │
  │  Fuel prices       →    GBM-based fuel cost simulation          │
  └──────────────────────────────────────────────────────────────────┘

Outputs:
  - Port congestion matrix over time
  - Vessel waiting time distributions
  - Dynamic route cost adjustments
  - Disruption event log
  - Berth utilisation time series
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


# ────────────────────────────────────────────────────────────────────
# Event Classes
# ────────────────────────────────────────────────────────────────────

@dataclass
class MaritimeEvent:
    event_type:  str      # weather | congestion | breakdown | canal_closure | strike | fog
    port_id:     str
    severity:    float    # 0-1
    duration_hr: float    # hours
    timestamp:   datetime = field(default_factory=datetime.now)

    EVENT_IMPACTS = {
        "weather":        {"delay_mult": 2.5, "throughput_loss": 0.40},
        "congestion":     {"delay_mult": 1.8, "throughput_loss": 0.25},
        "breakdown":      {"delay_mult": 3.0, "throughput_loss": 0.60},
        "canal_closure":  {"delay_mult": 5.0, "throughput_loss": 0.80},
        "strike":         {"delay_mult": 4.0, "throughput_loss": 0.70},
        "fog":            {"delay_mult": 1.5, "throughput_loss": 0.15},
    }

    @property
    def delay_hours(self) -> float:
        base = self.EVENT_IMPACTS.get(self.event_type, {}).get("delay_mult", 1.0)
        return self.severity * base * self.duration_hr

    @property
    def throughput_factor(self) -> float:
        loss = self.EVENT_IMPACTS.get(self.event_type, {}).get("throughput_loss", 0.0)
        return max(0.1, 1.0 - self.severity * loss)

    @property
    def cost_multiplier(self) -> float:
        return 1.0 + self.severity * 0.8   # up to 80% cost increase


# ────────────────────────────────────────────────────────────────────
# Digital Twin Core
# ────────────────────────────────────────────────────────────────────

class PortDigitalTwin:
    """
    Digital Twin for a global maritime port network.
    Simulates real-time operational state and propagates
    disruptions into QUBO cost adjustments.
    """

    # Weather state transition matrix (Markov chain)
    WEATHER_STATES  = ["Clear", "Cloudy", "Rain", "Storm", "Fog"]
    WEATHER_TRANS   = np.array([
        [0.70, 0.15, 0.08, 0.04, 0.03],   # Clear
        [0.20, 0.50, 0.18, 0.07, 0.05],   # Cloudy
        [0.15, 0.25, 0.45, 0.10, 0.05],   # Rain
        [0.10, 0.20, 0.30, 0.30, 0.10],   # Storm
        [0.30, 0.20, 0.15, 0.05, 0.30],   # Fog
    ])

    ARRIVAL_RATES = {   # vessels/day per port
        "SHA": 18, "SIN": 15, "HKG": 12, "ROT": 16,
        "HAM": 10, "ANT":  9, "LAX": 14, "NYK": 11,
        "DXB": 13, "COL":  7, "MOM":  5, "SAO":  8,
    }

    def __init__(self, port_ids: List[str], seed: int = 2026):
        self.port_ids    = port_ids
        self.rng         = np.random.default_rng(seed)
        self.events:     List[MaritimeEvent] = []
        self.weather_state = {p: 0 for p in port_ids}  # 0=Clear
        self.berth_state   = {p: 0.3 for p in port_ids}  # initial 30% occupancy
        self.fuel_price    = 650.0   # $/tonne starting price

    # ── Weather Simulation (Markov chain) ────────────────────────────

    def simulate_weather(self, n_hours: int = 72) -> pd.DataFrame:
        records = []
        states  = {p: 0 for p in self.port_ids}

        for h in range(n_hours):
            row = {"hour": h}
            for p in self.port_ids:
                trans   = self.WEATHER_TRANS[states[p]]
                new_st  = int(self.rng.choice(len(self.WEATHER_STATES), p=trans))
                states[p] = new_st
                row[p]  = self.WEATHER_STATES[new_st]
            records.append(row)

        return pd.DataFrame(records).set_index("hour")

    # ── Vessel Arrival Process (Poisson) ────────────────────────────

    def simulate_arrivals(self, n_days: int = 7) -> pd.DataFrame:
        records = []
        base_date = datetime(2026, 3, 20)

        for p in self.port_ids:
            lam = self.ARRIVAL_RATES.get(p, 8)   # vessels/day
            for day in range(n_days):
                n_arr = self.rng.poisson(lam)
                for _ in range(n_arr):
                    hour = self.rng.uniform(0, 24)
                    records.append({
                        "port":      p,
                        "day":       day,
                        "hour":      round(float(hour), 2),
                        "timestamp": base_date + timedelta(days=day, hours=float(hour)),
                        "vessel_type": self.rng.choice(["ULCV","VLCV","Feeder","Tanker"],
                                                        p=[0.2,0.3,0.35,0.15]),
                    })

        return pd.DataFrame(records).sort_values("timestamp")

    # ── M/M/c Berth Queue Simulation ────────────────────────────────

    def simulate_berth_queue(
        self, port_id: str, n_berths: int, n_hours: int = 48
    ) -> pd.DataFrame:
        """
        Simulate berth occupancy via discrete-event M/M/c queue.
        λ = arrival rate, μ = service rate (berths).
        """
        lam   = self.ARRIVAL_RATES.get(port_id, 8) / 24   # vessels/hour
        mu    = 1 / 6    # mean 6hr berth time → μ=1/6 per berth
        rho   = lam / (n_berths * mu)

        records   = []
        occupied  = 0
        wait_time = 0.0

        for h in range(n_hours):
            # Arrivals this hour
            arrivals  = self.rng.poisson(lam)
            # Departures
            departures = min(occupied, self.rng.poisson(mu * max(occupied, 1)))
            occupied  = max(0, occupied + arrivals - departures)
            occupied  = min(occupied, n_berths)
            util      = occupied / n_berths
            # Waiting vessels
            queue     = max(0, occupied + arrivals - n_berths)
            wait_time = queue / (mu * n_berths + 1e-6)

            records.append({
                "hour":           h,
                "occupied_berths":occupied,
                "utilisation":    round(util, 3),
                "queue_length":   queue,
                "wait_time_hr":   round(wait_time, 2),
            })

        return pd.DataFrame(records).set_index("hour")

    # ── Tidal Model ─────────────────────────────────────────────────

    def simulate_tides(self, port_id: str, n_hours: int = 48) -> np.ndarray:
        """Sinusoidal tidal model with M2 (12.42h) + S2 (12h) components."""
        t     = np.linspace(0, n_hours, n_hours)
        phase = self.rng.uniform(0, 2*np.pi)
        tide  = (2.5 * np.sin(2*np.pi*t/12.42 + phase) +
                 0.8 * np.sin(2*np.pi*t/12.0  + phase*0.7) +
                 0.2 * self.rng.standard_normal(n_hours))
        return tide

    # ── Fuel Price Simulation (GBM) ──────────────────────────────────

    def simulate_fuel_prices(self, n_days: int = 30) -> np.ndarray:
        """Geometric Brownian Motion for bunker fuel prices."""
        mu_gbm = 0.0002   # drift
        sigma  = 0.015    # volatility
        prices = [self.fuel_price]
        for _ in range(n_days - 1):
            dW = self.rng.standard_normal()
            prices.append(prices[-1] * np.exp((mu_gbm - 0.5*sigma**2) + sigma*dW))
        return np.array(prices)

    # ── Disruption Event Generator ───────────────────────────────────

    def generate_disruptions(
        self, scenario: str = "normal", n_days: int = 7
    ) -> List[MaritimeEvent]:
        SCENARIO_PROBS = {
            "normal":  {"weather":0.10,"congestion":0.08,"breakdown":0.03,"canal_closure":0.01,"strike":0.02,"fog":0.06},
            "stressed":{"weather":0.20,"congestion":0.18,"breakdown":0.08,"canal_closure":0.03,"strike":0.06,"fog":0.12},
            "crisis":  {"weather":0.40,"congestion":0.35,"breakdown":0.15,"canal_closure":0.20,"strike":0.15,"fog":0.20},
        }
        probs  = SCENARIO_PROBS.get(scenario, SCENARIO_PROBS["normal"])
        events = []
        base   = datetime(2026, 3, 20)

        for day in range(n_days):
            for etype, prob in probs.items():
                if self.rng.random() < prob:
                    port = self.rng.choice(self.port_ids)
                    ev   = MaritimeEvent(
                        event_type  = etype,
                        port_id     = port,
                        severity    = float(self.rng.uniform(0.2, 0.95)),
                        duration_hr = float(self.rng.uniform(2, 48)),
                        timestamp   = base + timedelta(days=day,
                                        hours=float(self.rng.uniform(0,24))),
                    )
                    events.append(ev)

        self.events = events
        return events

    # ── Cost Adjustment from Events ──────────────────────────────────

    def compute_route_cost_multipliers(
        self, routes: Dict, events: Optional[List[MaritimeEvent]] = None
    ) -> Dict:
        """
        For each vessel route, compute cost multiplier based on
        disruption events at the ports on the route.
        """
        if events is None:
            events = self.events

        # Build per-port cost multiplier
        port_mults = {p: 1.0 for p in self.port_ids}
        for ev in events:
            p = ev.port_id
            port_mults[p] = max(port_mults[p], ev.cost_multiplier)

        # Apply to routes
        adjusted = {}
        for vi, vessel_routes in routes.items():
            adjusted[vi] = []
            for r in vessel_routes:
                ports_on_route = set(r["route"])
                max_mult = max(port_mults.get(p, 1.0) for p in ports_on_route)
                adj_cost = r["total_cost"] * max_mult
                adjusted[vi].append({**r, "total_cost": adj_cost, "cost_multiplier": max_mult})
        return adjusted

    # ── Port Throughput Matrix ────────────────────────────────────────

    def compute_throughput_matrix(
        self, events: Optional[List[MaritimeEvent]] = None
    ) -> pd.DataFrame:
        """
        Returns current throughput factor (0-1) for each port,
        accounting for active disruptions.
        """
        if events is None:
            events = self.events

        throughput = {p: 1.0 for p in self.port_ids}
        for ev in events:
            p = ev.port_id
            throughput[p] = min(throughput[p], ev.throughput_factor)

        return pd.Series(throughput, name="throughput_factor").to_frame()

    # ── Congestion Heatmap ────────────────────────────────────────────

    def simulate_congestion_heatmap(self, n_hours: int = 48) -> pd.DataFrame:
        records = []
        for h in range(n_hours):
            row = {"hour": h}
            for p in self.port_ids:
                base   = 0.40 + 0.25 * np.sin(h * np.pi / 12)
                noise  = 0.05 * self.rng.standard_normal()
                # Add event impact
                event_factor = max(
                    (1 - ev.throughput_factor) * np.exp(-abs(h - 12)/10)
                    for ev in self.events if ev.port_id == p
                ) if any(ev.port_id == p for ev in self.events) else 0.0
                cong = float(np.clip(base + noise + event_factor * 0.3, 0.05, 1.0))
                row[p] = round(cong, 3)
            records.append(row)
        return pd.DataFrame(records).set_index("hour")

    # ── Event Summary ─────────────────────────────────────────────────

    def get_event_summary(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp":    ev.timestamp.strftime("%Y-%m-%d %H:%M"),
            "port":         ev.port_id,
            "event_type":   ev.event_type,
            "severity":     round(ev.severity, 3),
            "duration_hr":  round(ev.duration_hr, 1),
            "delay_hours":  round(ev.delay_hours, 1),
            "throughput%":  round(ev.throughput_factor * 100, 1),
            "cost_mult":    round(ev.cost_multiplier, 3),
        } for ev in self.events]).sort_values("delay_hours", ascending=False)

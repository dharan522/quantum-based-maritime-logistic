"""
generate_data.py
Generates synthetic Maritime Logistics data:
  - Global port network (nodes + coordinates)
  - Vessel fleet (capacity, speed, fuel)
  - Cargo manifest (origin, destination, weight, priority, deadline)
  - Shipping lane distances and travel times
  - Port berth capacities and handling rates
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

np.random.seed(2026)

# ── Port Network ────────────────────────────────────────────────────────────
PORTS = {
    "SHA": {"name": "Shanghai",       "lat": 31.23,  "lon": 121.47, "berths": 8,  "handling_rate": 4500, "region": "Asia"},
    "SIN": {"name": "Singapore",      "lat":  1.29,  "lon": 103.85, "berths": 6,  "handling_rate": 4200, "region": "Asia"},
    "HKG": {"name": "Hong Kong",      "lat": 22.30,  "lon": 114.17, "berths": 5,  "handling_rate": 3800, "region": "Asia"},
    "ROT": {"name": "Rotterdam",      "lat": 51.92,  "lon":   4.48, "berths": 10, "handling_rate": 5200, "region": "Europe"},
    "HAM": {"name": "Hamburg",        "lat": 53.55,  "lon":   9.99, "berths": 7,  "handling_rate": 3900, "region": "Europe"},
    "ANT": {"name": "Antwerp",        "lat": 51.22,  "lon":   4.40, "berths": 6,  "handling_rate": 4100, "region": "Europe"},
    "LAX": {"name": "Los Angeles",    "lat": 33.73,  "lon": -118.27,"berths": 9,  "handling_rate": 4700, "region": "Americas"},
    "NYK": {"name": "New York",       "lat": 40.66,  "lon": -74.04, "berths": 7,  "handling_rate": 4000, "region": "Americas"},
    "DXB": {"name": "Dubai (Jebel Ali)","lat":24.99,  "lon":  55.06, "berths": 8,  "handling_rate": 4600, "region": "Middle East"},
    "COL": {"name": "Colombo",        "lat":  6.93,  "lon":  79.85, "berths": 5,  "handling_rate": 3500, "region": "Asia"},
    "MOM": {"name": "Mombasa",        "lat": -4.05,  "lon":  39.66, "berths": 4,  "handling_rate": 2800, "region": "Africa"},
    "SAO": {"name": "Santos",         "lat": -23.95, "lon": -46.33, "berths": 6,  "handling_rate": 3600, "region": "Americas"},
}

PORT_IDS = list(PORTS.keys())
N_PORTS  = len(PORT_IDS)

# ── Vessels ─────────────────────────────────────────────────────────────────
VESSELS = [
    {"id":"V01","name":"Pacific Titan",  "type":"ULCV","capacity_teu":20000,"speed_knots":22,"fuel_tday":280,"cost_day":45000},
    {"id":"V02","name":"Atlantic Star",  "type":"ULCV","capacity_teu":18000,"speed_knots":21,"fuel_tday":265,"cost_day":42000},
    {"id":"V03","name":"Indian Express", "type":"VLCV","capacity_teu":14000,"speed_knots":20,"fuel_tday":210,"cost_day":34000},
    {"id":"V04","name":"Suez Carrier",   "type":"VLCV","capacity_teu":12000,"speed_knots":19,"fuel_tday":190,"cost_day":30000},
    {"id":"V05","name":"Feeder Alpha",   "type":"Feeder","capacity_teu":4000,"speed_knots":17,"fuel_tday":95, "cost_day":14000},
    {"id":"V06","name":"Feeder Beta",    "type":"Feeder","capacity_teu":3500,"speed_knots":16,"fuel_tday":85, "cost_day":12000},
    {"id":"V07","name":"Coastal Runner", "type":"Feeder","capacity_teu":2000,"speed_knots":15,"fuel_tday":55, "cost_day":8000},
    {"id":"V08","name":"Tanker Zeus",    "type":"Tanker","capacity_teu":8000,"speed_knots":18,"fuel_tday":145,"cost_day":22000},
]

# ── Distance Matrix (nautical miles, approximate great-circle) ──────────────
def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # Earth radius in nautical miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_distance_matrix():
    n = N_PORTS
    D = np.zeros((n, n))
    for i, p1 in enumerate(PORT_IDS):
        for j, p2 in enumerate(PORT_IDS):
            if i != j:
                D[i,j] = haversine_nm(
                    PORTS[p1]["lat"], PORTS[p1]["lon"],
                    PORTS[p2]["lat"], PORTS[p2]["lon"]
                )
    return D

# ── Cargo Manifest ───────────────────────────────────────────────────────────
def generate_cargo(n_cargoes=40):
    rng = np.random.default_rng(2026)
    records = []
    types   = ["Container","Bulk","Liquid","Ro-Ro","Refrigerated"]
    priorities = ["CRITICAL","HIGH","MEDIUM","LOW"]
    base_date  = datetime(2026, 3, 20)

    for i in range(n_cargoes):
        origin = rng.choice(PORT_IDS)
        dest   = rng.choice([p for p in PORT_IDS if p != origin])
        weight = int(rng.integers(200, 5000))   # TEUs
        ptype  = rng.choice(types,   p=[0.5,0.2,0.15,0.1,0.05])
        prio   = rng.choice(priorities, p=[0.1,0.25,0.4,0.25])
        deadline_days = {"CRITICAL":5,"HIGH":10,"MEDIUM":20,"LOW":30}[prio]
        deadline = base_date + timedelta(days=int(rng.integers(deadline_days, deadline_days*2)))
        revenue  = weight * rng.uniform(80, 250)

        records.append({
            "cargo_id":   f"C{i+1:03d}",
            "origin":     origin,
            "destination":dest,
            "weight_teu": weight,
            "type":       ptype,
            "priority":   prio,
            "deadline":   deadline.strftime("%Y-%m-%d"),
            "revenue_usd":round(revenue, 2),
        })
    return pd.DataFrame(records)

# ── Port Congestion Time Series ──────────────────────────────────────────────
def generate_congestion(n_steps=72):
    rng = np.random.default_rng(999)
    records = []
    for step in range(n_steps):
        row = {"hour": step}
        for pid in PORT_IDS:
            base = 0.45 + 0.25*np.sin(step*np.pi/24) + 0.05*rng.standard_normal()
            spike = rng.exponential(0.1) if rng.random() < 0.08 else 0.0
            row[pid] = float(np.clip(base + spike, 0.1, 1.0))
        records.append(row)
    return pd.DataFrame(records).set_index("hour")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)

    # Distance matrix
    D = build_distance_matrix()
    dist_df = pd.DataFrame(D, index=PORT_IDS, columns=PORT_IDS)
    dist_df.to_csv("data/distance_matrix.csv")

    # Travel time (hours) for each vessel type
    for v in VESSELS[:3]:
        spd = v["speed_knots"]
        T   = D / spd
        pd.DataFrame(T, index=PORT_IDS, columns=PORT_IDS)\
          .to_csv(f"data/travel_time_{v['id']}.csv")

    # Port data
    port_df = pd.DataFrame(PORTS).T
    port_df.index.name = "port_id"
    port_df.to_csv("data/ports.csv")

    # Vessels
    pd.DataFrame(VESSELS).to_csv("data/vessels.csv", index=False)

    # Cargo
    cargo_df = generate_cargo(40)
    cargo_df.to_csv("data/cargo.csv", index=False)

    # Congestion
    cong_df = generate_congestion(72)
    cong_df.to_csv("data/congestion.csv")

    # JSON for dashboard
    with open("data/ports.json","w") as f:
        json.dump(PORTS, f, indent=2)

    print("✅ Maritime data generated:")
    print(f"   Ports    : {N_PORTS}")
    print(f"   Vessels  : {len(VESSELS)}")
    print(f"   Cargoes  : {len(cargo_df)}")
    print(f"   Distance range: {D[D>0].min():.0f} – {D.max():.0f} nm")
    print(f"   Files    : data/ports.csv, vessels.csv, cargo.csv,")
    print(f"              distance_matrix.csv, congestion.csv")

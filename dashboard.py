"""
dashboard.py
Interactive Dash Dashboard — QAOA Maritime Logistics Optimiser + Digital Twin

Run:  python dashboard.py
Open: http://127.0.0.1:8051
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash.dash_table as dt

from src.qubo_formulator  import MaritimeQUBOFormulator
from src.qaoa_solver      import MaritimeQAOASolver
from src.classical_solver import ClassicalMaritimeSolver
from src.digital_twin     import PortDigitalTwin

# ── Colour Palette ─────────────────────────────────────────────────────────
C = {
    "bg":      "#050E1A",
    "panel":   "#071D2E",
    "card":    "#0A2540",
    "border":  "#143252",
    "teal":    "#00C9C8",
    "cyan":    "#00E5FF",
    "amber":   "#FFB347",
    "coral":   "#FF6B6B",
    "green":   "#2ECC71",
    "purple":  "#C39BD3",
    "text":    "#D6EAF8",
    "muted":   "#7F8C8D",
}

PLOTLY_BASE = dict(
    paper_bgcolor=C["card"],
    plot_bgcolor=C["panel"],
    font=dict(color=C["text"], family="'Courier New', monospace", size=11),
    margin=dict(l=48, r=20, t=44, b=36),
    xaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False),
)

def load_data():
    dist  = pd.read_csv("data/distance_matrix.csv", index_col=0)
    ports = pd.read_csv("data/ports.csv", index_col=0)
    vessels = pd.read_csv("data/vessels.csv")
    cargo   = pd.read_csv("data/cargo.csv")
    with open("data/ports.json") as f:
        ports_json = json.load(f)
    return dist, ports, vessels, cargo, ports_json

# ── UI Helpers ──────────────────────────────────────────────────────────────

def card(children, style_extra=None):
    s = {"background": C["card"], "border": f"1px solid {C['border']}",
         "borderRadius": "8px", "padding": "18px", "marginBottom": "14px"}
    if style_extra: s.update(style_extra)
    return html.Div(children, style=s)

def lbl(text):
    return html.Label(text, style={"color": C["muted"], "fontSize": "11px",
                                    "marginBottom": "4px", "display": "block",
                                    "letterSpacing": "0.08em", "textTransform": "uppercase"})

def kpi(title, val, color=None, unit=""):
    return html.Div([
        html.Div(title, style={"color": C["muted"], "fontSize": "10px",
                               "letterSpacing": "0.1em", "textTransform": "uppercase"}),
        html.Div([
            html.Span(val,  style={"color": color or C["cyan"], "fontSize": "20px",
                                   "fontWeight": "700", "fontFamily": "Courier New, monospace"}),
            html.Span(unit, style={"color": C["muted"], "fontSize": "11px", "marginLeft": "4px"}),
        ], style={"marginTop": "2px"}),
    ], style={
        "background": C["panel"], "border": f"1px solid {C['border']}",
        "borderLeft": f"3px solid {color or C['cyan']}",
        "borderRadius": "6px", "padding": "12px 16px",
        "flex": "1", "minWidth": "120px",
    })

def badge(text, color):
    return html.Span(text, style={
        "background": C["panel"], "border": f"1px solid {color}",
        "color": color, "borderRadius": "4px", "padding": "3px 8px",
        "fontSize": "10px", "letterSpacing": "0.08em", "marginLeft": "6px",
    })

# ── App Layout ──────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "⚓ QAOA Maritime Logistics"

dist, ports, vessels, cargo, ports_json = load_data()
PORT_IDS = list(dist.index)

app.layout = html.Div(style={"background": C["bg"], "minHeight": "100vh",
                              "fontFamily": "'Courier New', monospace", "color": C["text"]}, children=[

    # ── HEADER ─────────────────────────────────────────────────────────
    html.Div(style={
        "background": C["panel"], "borderBottom": f"2px solid {C['teal']}22",
        "padding": "14px 28px", "display": "flex", "alignItems": "center", "gap": "14px",
    }, children=[
        html.Div("⚓", style={"fontSize": "30px"}),
        html.Div([
            html.H1("QAOA Maritime Logistics Optimiser",
                    style={"margin": "0", "fontSize": "18px", "color": C["cyan"],
                           "letterSpacing": "0.05em"}),
            html.Div("Quantum Approximate Optimisation  ·  Digital Twin Port Simulation  ·  Route & Berth Optimisation",
                     style={"color": C["muted"], "fontSize": "11px", "marginTop": "2px"}),
        ]),
        html.Div(style={"marginLeft": "auto", "display": "flex", "gap": "6px"}, children=[
            badge("QISKIT 2.3", C["cyan"]),
            badge("AER STATEVECTOR", C["teal"]),
            badge("22-QUBIT QUBO", C["amber"]),
        ]),
    ]),

    # ── BODY ───────────────────────────────────────────────────────────
    html.Div(style={"display": "flex", "minHeight": "calc(100vh - 65px)"}, children=[

        # ── SIDEBAR ────────────────────────────────────────────────────
        html.Div(style={
            "width": "270px", "minWidth": "270px",
            "background": C["panel"], "borderRight": f"1px solid {C['border']}",
            "padding": "18px 14px", "overflowY": "auto",
        }, children=[

            html.H3("⚙  Parameters", style={"margin": "0 0 14px", "fontSize": "13px",
                                              "color": C["purple"], "letterSpacing": "0.1em"}),

            lbl("QAOA Layers (p)"),
            dcc.Slider(1, 4, 1, value=2, id="sl-layers",
                marks={i: {"label": str(i), "style": {"color": C["text"], "fontSize": "11px"}}
                       for i in range(1, 5)},
                tooltip={"placement": "top"}),

            html.Div(style={"height": "14px"}),
            lbl("Number of Vessels"),
            dcc.Slider(2, 8, 1, value=4, id="sl-vessels",
                marks={i: {"label": str(i), "style": {"color": C["text"], "fontSize": "11px"}}
                       for i in range(2, 9)},
                tooltip={"placement": "top"}),

            html.Div(style={"height": "14px"}),
            lbl("Routes per Vessel"),
            dcc.Slider(2, 4, 1, value=3, id="sl-routes",
                marks={i: {"label": str(i), "style": {"color": C["text"], "fontSize": "11px"}}
                       for i in range(2, 5)},
                tooltip={"placement": "top"}),

            html.Div(style={"height": "14px"}),
            lbl("Shots"),
            dcc.Dropdown(id="dd-shots",
                options=[{"label": str(s), "value": s} for s in [512, 1024, 2048]],
                value=1024, clearable=False,
                style={"background": C["panel"], "color": C["bg"],
                       "border": f"1px solid {C['border']}"}),

            html.Div(style={"height": "16px"}),
            lbl("Digital Twin Scenario"),
            dcc.RadioItems(id="rd-scenario",
                options=[
                    {"label": "  🟢  Normal",   "value": "normal"},
                    {"label": "  🟡  Stressed",  "value": "stressed"},
                    {"label": "  🔴  Crisis",    "value": "crisis"},
                ],
                value="stressed",
                labelStyle={"display": "block", "margin": "7px 0", "cursor": "pointer"},
                style={"color": C["text"], "fontSize": "13px"},
            ),

            html.Div(style={"height": "20px"}),
            html.Button("▶  RUN QAOA", id="btn-run", n_clicks=0, style={
                "width": "100%", "padding": "13px",
                "background": f"linear-gradient(135deg, {C['teal']}, {C['cyan']})",
                "color": C["bg"], "border": "none", "borderRadius": "6px",
                "fontSize": "13px", "fontWeight": "bold", "cursor": "pointer",
                "fontFamily": "Courier New, monospace", "letterSpacing": "0.08em",
            }),
            html.Div(id="run-status", style={"color": C["muted"], "fontSize": "10px",
                                              "textAlign": "center", "marginTop": "6px"}),

            html.Div(style={"height": "24px"}),
            html.Hr(style={"borderColor": C["border"]}),
            html.H3("🚢  Fleet", style={"fontSize": "12px", "color": C["purple"],
                                         "margin": "10px 0 8px", "letterSpacing": "0.1em"}),
            html.Div([
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between",
                    "padding": "4px 0", "borderBottom": f"1px solid {C['border']}",
                }, children=[
                    html.Span(v["name"][:12], style={"color": C["text"], "fontSize": "10px"}),
                    html.Span(v["type"],      style={"color": C["teal"],  "fontSize": "9px"}),
                    html.Span(f"{v['capacity_teu']:,} TEU",
                              style={"color": C["amber"], "fontSize": "9px"}),
                ])
                for _, v in vessels.iterrows()
            ]),

            html.Div(style={"height": "16px"}),
            html.H3("🌍  Port Universe", style={"fontSize": "12px", "color": C["purple"],
                                                  "margin": "10px 0 8px", "letterSpacing": "0.1em"}),
            html.Div([
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between",
                    "padding": "3px 0", "borderBottom": f"1px solid {C['border']}",
                }, children=[
                    html.Span(pid,          style={"color": C["cyan"],  "fontSize": "10px"}),
                    html.Span(PORTS_DATA[pid]["name"][:14] if pid in PORTS_DATA else "",
                              style={"color": C["text"],  "fontSize": "10px"}),
                ])
                for pid in PORT_IDS
            ]),
        ]),

        # ── MAIN CONTENT ───────────────────────────────────────────────
        html.Div(style={"flex": "1", "padding": "18px", "overflowY": "auto"}, children=[
            dcc.Loading(color=C["cyan"], type="dot", children=[

                # KPI Row
                html.Div(id="kpi-row", style={"display": "flex", "gap": "10px",
                                               "marginBottom": "14px", "flexWrap": "wrap"}),

                # Row 1: Map + Convergence
                html.Div(style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr",
                                "gap": "14px", "marginBottom": "14px"}, children=[
                    card([html.H4("🌐  Global Port Network + QAOA Route Assignment",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-map", config={"displayModeBar": False})]),
                    card([html.H4("📈  QAOA Convergence",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-conv", config={"displayModeBar": False})]),
                ]),

                # Row 2: Digital Twin congestion + berth queue
                card([html.H4("🔮  Digital Twin — Port Congestion Heatmap (48h)",
                               style={"margin": "0 0 8px", "fontSize": "12px",
                                      "color": C["purple"], "letterSpacing": "0.08em"}),
                      dcc.Graph(id="fig-cong", config={"displayModeBar": False})]),

                # Row 3: Berth Queue + Fuel
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                "gap": "14px", "marginBottom": "14px"}, children=[
                    card([html.H4("⚓  M/M/c Berth Queue — Shanghai Port",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-berth", config={"displayModeBar": False})]),
                    card([html.H4("⛽  Bunker Fuel Price Simulation (GBM)",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-fuel", config={"displayModeBar": False})]),
                ]),

                # Row 4: Energy landscape + QUBO heatmap
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                "gap": "14px", "marginBottom": "14px"}, children=[
                    card([html.H4("🗺  QAOA Energy Landscape (p=1 slice)",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-land", config={"displayModeBar": False})]),
                    card([html.H4("🧮  QUBO Matrix Heatmap",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-qubo", config={"displayModeBar": False})]),
                ]),

                # Row 5: Probability dist + benchmark
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                "gap": "14px", "marginBottom": "14px"}, children=[
                    card([html.H4("📊  Measurement Probability Distribution",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-prob", config={"displayModeBar": False})]),
                    card([html.H4("⚖  QAOA vs Classical Cost Benchmark",
                                  style={"margin": "0 0 8px", "fontSize": "12px",
                                         "color": C["purple"], "letterSpacing": "0.08em"}),
                          dcc.Graph(id="fig-bench", config={"displayModeBar": False})]),
                ]),

                # Event table
                card([html.H4("📋  Digital Twin — Disruption Event Log",
                               style={"margin": "0 0 10px", "fontSize": "12px",
                                      "color": C["purple"], "letterSpacing": "0.08em"}),
                      html.Div(id="event-table")]),

                # Route table
                card([html.H4("🗺  Vessel Route Candidates + QAOA Assignments",
                               style={"margin": "0 0 10px", "fontSize": "12px",
                                      "color": C["purple"], "letterSpacing": "0.08em"}),
                      html.Div(id="route-table")]),
            ]),
        ]),
    ]),

    dcc.Store(id="store"),
])

PORTS_DATA = ports_json

# ── Main Callback ──────────────────────────────────────────────────────────

@app.callback(
    Output("store", "data"),
    Output("run-status", "children"),
    Input("btn-run", "n_clicks"),
    State("sl-layers",  "value"),
    State("sl-vessels", "value"),
    State("sl-routes",  "value"),
    State("dd-shots",   "value"),
    State("rd-scenario","value"),
    prevent_initial_call=False,
)
def run_qaoa(n_clicks, p_layers, n_vessels, n_routes, shots, scenario):
    dist_l, ports_l, vessels_l, cargo_l, _ = load_data()
    port_ids = list(dist_l.index)

    # Digital Twin
    twin    = PortDigitalTwin(port_ids, seed=2026)
    events  = twin.generate_disruptions(scenario=scenario, n_days=7)
    cong    = twin.simulate_congestion_heatmap(n_hours=48)
    berth_Q = twin.simulate_berth_queue("SHA", n_berths=8, n_hours=48)
    fuel_px = twin.simulate_fuel_prices(n_days=30).tolist()
    arrivals= twin.simulate_arrivals(n_days=7)
    ev_df   = twin.get_event_summary()
    tp      = twin.compute_throughput_matrix(events)

    # QUBO
    fml  = MaritimeQUBOFormulator(dist_l, vessels_l, cargo_l, ports_l,
                                   n_vessels=n_vessels, n_routes=n_routes)
    adj  = twin.compute_route_cost_multipliers(fml.routes, events)
    fml.routes = adj
    Q, meta = fml.build_qubo()

    # QAOA
    solver = MaritimeQAOASolver(Q, meta["var_labels"], p_layers=p_layers, shots=shots)
    result = solver.optimise(maxiter=100)
    qaoa_eval = fml.evaluate_solution(result["best_bitstring"])
    top_k     = solver.top_k_solutions(k=10)
    routes_df = fml.get_route_summary()

    # Landscape (small sub-QUBO)
    sub_n = min(6, Q.shape[0])
    sl1 = MaritimeQAOASolver(Q[:sub_n,:sub_n], meta["var_labels"][:sub_n], p_layers=1, shots=128)
    g, b, E = sl1.scan_landscape(resolution=14)

    # Classical
    classical = ClassicalMaritimeSolver(dist_l, vessels_l, cargo_l, ports_l)
    greedy    = classical.greedy_route_assignment(n_vessels)
    cw_res    = classical.clarke_wright_savings(n_vessels=n_vessels)
    bench     = classical.benchmark_summary(qaoa_eval["total_cost"], n_vessels)

    # Build assigned routes list for map
    assigned_routes_map = []
    for vi in range(fml.n_v):
        for ri in range(fml.n_r):
            key = f"x_v{vi}_r{ri}"
            if key in meta["var_labels"]:
                qi = meta["var_labels"].index(key)
                bs = result["best_bitstring"]
                if qi < len(bs) and bs[qi] == "1":
                    if ri < len(fml.routes[vi]):
                        assigned_routes_map.append({
                            "vessel_idx": vi,
                            "route": list(fml.routes[vi][ri]["route"]),
                            "cost":  fml.routes[vi][ri]["total_cost"],
                        })

    return {
        "qaoa_eval":      qaoa_eval,
        "energy_history": solver.energy_history,
        "optimal_energy": solver.optimal_energy,
        "counts":         result["counts"],
        "best_bs":        result["best_bitstring"],
        "top_k":          top_k,
        "Q":              Q.tolist(),
        "meta":           {**meta, "var_labels": meta["var_labels"]},
        "landscape_g":    g.tolist(),
        "landscape_b":    b.tolist(),
        "landscape_E":    E.tolist(),
        "cong":           cong.to_dict(),
        "berth":          berth_Q.to_dict(),
        "fuel":           fuel_px,
        "arrivals":       arrivals.groupby("port").size().to_dict(),
        "throughput":     tp["throughput_factor"].to_dict(),
        "events":         ev_df.to_dict("records") if not ev_df.empty else [],
        "routes_df":      routes_df.to_dict("records"),
        "bench":          bench.to_dict("records"),
        "greedy_cost":    greedy["total_cost"],
        "cw_cost":        cw_res["total_cost"],
        "assigned_routes":assigned_routes_map,
        "scenario":       scenario,
        "n_vessels":      n_vessels,
        "p_layers":       p_layers,
    }, f"✓ Done  p={p_layers}  {len(solver.energy_history)} iters"


# ── Chart Callbacks ────────────────────────────────────────────────────────

@app.callback(Output("kpi-row","children"), Input("store","data"))
def cb_kpi(d):
    if not d: return []
    qe = d["qaoa_eval"]
    return [
        kpi("QAOA Energy",     f"{qe['qubo_energy']:.3f}", C["cyan"]),
        kpi("Total Route Cost",f"${qe['total_cost']:,.0f}", C["amber"]),
        kpi("Vessels Assigned",f"{qe['n_assigned']}/{d['n_vessels']}", C["green"]),
        kpi("Constraints",     "✅ OK" if qe["constraints_satisfied"] else "❌ FAIL",
            C["green"] if qe["constraints_satisfied"] else C["coral"]),
        kpi("QAOA Layers",     str(d["p_layers"]), C["purple"]),
        kpi("Scenario",        d["scenario"].upper(), C["teal"]),
    ]


@app.callback(Output("fig-map","figure"), Input("store","data"))
def cb_map(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    fig = go.Figure(layout={**PLOTLY_BASE,
        "geo": dict(bgcolor=C["panel"], showland=True, landcolor="#0D2035",
                    showocean=True, oceancolor="#071520",
                    showcoastlines=True, coastlinecolor=C["border"],
                    projection_type="natural earth"),
        "paper_bgcolor": C["card"], "margin": dict(l=0,r=0,t=30,b=0),
    })
    # All ports
    lats = [ports_json[p]["lat"] for p in PORT_IDS]
    lons = [ports_json[p]["lon"] for p in PORT_IDS]
    names= [ports_json[p]["name"] for p in PORT_IDS]
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, text=names,
        mode="markers+text", textposition="top right",
        textfont=dict(size=9, color=C["muted"]),
        marker=dict(size=8, color=C["teal"], opacity=0.8,
                    line=dict(width=1, color=C["cyan"])),
        name="Ports"))

    # Assigned routes
    colours = [C["cyan"], C["amber"], C["coral"], C["green"]]
    for i, ar in enumerate(d.get("assigned_routes", [])):
        route = ar["route"]
        col   = colours[i % len(colours)]
        rlats = [ports_json[p]["lat"] for p in route if p in ports_json]
        rlons = [ports_json[p]["lon"] for p in route if p in ports_json]
        fig.add_trace(go.Scattergeo(
            lat=rlats, lon=rlons, mode="lines+markers",
            line=dict(width=2, color=col),
            marker=dict(size=6, color=col),
            name=f"V{ar['vessel_idx']+1} Route",
        ))
    fig.update_layout(showlegend=True, height=340,
        legend=dict(bgcolor=C["card"], bordercolor=C["border"],
                    font=dict(size=9), x=0, y=0))
    return fig


@app.callback(Output("fig-conv","figure"), Input("store","data"))
def cb_conv(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    h   = d["energy_history"]
    opt = d["optimal_energy"]
    fig = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    fig.add_trace(go.Scatter(x=list(range(len(h))), y=h, mode="lines",
        line=dict(color=C["cyan"], width=1.8),
        fill="tozeroy", fillcolor=C["teal"]+"20", name="⟨HC⟩"))
    fig.add_hline(y=opt, line_color=C["coral"], line_dash="dash",
                  annotation_text=f"opt={opt:.3f}", annotation_font_color=C["coral"])
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Energy",
                      showlegend=False, height=340)
    return fig


@app.callback(Output("fig-cong","figure"), Input("store","data"))
def cb_cong(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    cong = pd.DataFrame(d["cong"])
    sub  = [p for p in PORT_IDS[:10] if p in cong.columns]
    fig  = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    fig.add_trace(go.Heatmap(
        z=cong[sub].values.T, x=cong.index.tolist(), y=sub,
        colorscale="YlOrRd", zmin=0, zmax=1,
        colorbar=dict(title="Congestion", thickness=12,
                      tickfont=dict(color=C["text"])),
    ))
    fig.update_layout(xaxis_title="Hour", height=220)
    return fig


@app.callback(Output("fig-berth","figure"), Input("store","data"))
def cb_berth(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    bq  = pd.DataFrame(d["berth"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(**{k:v for k,v in PLOTLY_BASE.items()
                         if k not in ["xaxis","yaxis"]}, height=260)
    fig.add_trace(go.Scatter(x=bq.index.tolist(), y=bq["utilisation"],
        mode="lines", line=dict(color=C["teal"], width=2),
        fill="tozeroy", fillcolor=C["teal"]+"18", name="Utilisation"), secondary_y=False)
    fig.add_trace(go.Scatter(x=bq.index.tolist(), y=bq["wait_time_hr"],
        mode="lines", line=dict(color=C["amber"], width=1.5, dash="dash"),
        name="Wait Time (hr)"), secondary_y=True)
    fig.update_xaxes(title_text="Hour", gridcolor=C["border"])
    fig.update_yaxes(title_text="Utilisation", gridcolor=C["border"], secondary_y=False)
    fig.update_yaxes(title_text="Wait (hr)",   gridcolor=C["border"], secondary_y=True)
    fig.update_layout(legend=dict(bgcolor=C["card"], bordercolor=C["border"]))
    return fig


@app.callback(Output("fig-fuel","figure"), Input("store","data"))
def cb_fuel(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    fp  = d["fuel"]
    fig = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    fig.add_trace(go.Scatter(x=list(range(len(fp))), y=fp, mode="lines",
        line=dict(color=C["amber"], width=2),
        fill="tozeroy", fillcolor=C["amber"]+"18", name="$/tonne"))
    fig.add_hline(y=650, line_color=C["muted"], line_dash="dot",
                  annotation_text="$650 baseline", annotation_font_color=C["muted"])
    fig.update_layout(xaxis_title="Day", yaxis_title="USD/tonne",
                      showlegend=False, height=260)
    return fig


@app.callback(Output("fig-land","figure"), Input("store","data"))
def cb_land(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    g, b, E = d["landscape_g"], d["landscape_b"], d["landscape_E"]
    fig = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    fig.add_trace(go.Contour(z=E, x=b, y=g, colorscale="Plasma",
        contours=dict(showlabels=False),
        colorbar=dict(thickness=10, tickfont=dict(color=C["text"]))))
    fig.update_layout(xaxis_title="β", yaxis_title="γ", height=260)
    return fig


@app.callback(Output("fig-qubo","figure"), Input("store","data"))
def cb_qubo(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    Q   = np.array(d["Q"])
    lbl = [l.replace("x_","").replace("b_","b:") for l in d["meta"]["var_labels"]]
    fig = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    fig.add_trace(go.Heatmap(z=Q, x=lbl, y=lbl, colorscale="RdBu", zmid=0,
        colorbar=dict(thickness=10, tickfont=dict(color=C["text"]))))
    fig.update_layout(xaxis_tickangle=-45, xaxis_tickfont_size=7,
                      yaxis_tickfont_size=7, height=260)
    return fig


@app.callback(Output("fig-prob","figure"), Input("store","data"))
def cb_prob(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    counts  = d["counts"]
    best_bs = d["best_bs"]
    total   = sum(counts.values())
    top18   = sorted(counts.items(), key=lambda x:x[1], reverse=True)[:18]
    labels  = [c[0][::-1] for c in top18]
    probs   = [c[1]/total for c in top18]
    cols    = [C["cyan"] if l==best_bs else C["teal"]+"AA" for l in labels]
    fig = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    fig.add_trace(go.Bar(x=list(range(len(labels))), y=probs,
        marker_color=cols, marker_line_width=0))
    fig.update_layout(xaxis=dict(tickvals=list(range(len(labels))),
                                  ticktext=labels, tickangle=-45, tickfont=dict(size=6)),
                      yaxis_title="Probability", height=260, showlegend=False)
    return fig


@app.callback(Output("fig-bench","figure"), Input("store","data"))
def cb_bench(d):
    if not d: return go.Figure(layout=PLOTLY_BASE)
    bdf  = pd.DataFrame(d["bench"])
    cols = [C["cyan"] if m=="QAOA" else C["amber"] if m=="Greedy-NN" else C["coral"]
            for m in bdf["method"]]
    vals = (bdf["total_cost"] / 1e6).tolist()
    fig  = go.Figure(layout={**PLOTLY_BASE, "title": ""})
    bars = fig.add_trace(go.Bar(x=bdf["method"], y=vals,
        marker_color=cols, marker_line_width=0, width=0.5))
    for i, v in enumerate(vals):
        fig.add_annotation(x=bdf["method"].iloc[i], y=v+0.02,
            text=f"${v:.2f}M", showarrow=False,
            font=dict(color=C["text"], size=10))
    fig.update_layout(yaxis_title="Total Cost (USD M)",
                      yaxis_range=[0, max(vals)*1.3], height=260, showlegend=False)
    return fig


@app.callback(Output("event-table","children"), Input("store","data"))
def cb_events(d):
    if not d or not d["events"]:
        return html.Div("No disruption events in this scenario.",
                        style={"color": C["muted"], "fontSize": "12px"})
    cols = ["timestamp","port","event_type","severity","duration_hr","delay_hours","throughput%","cost_mult"]
    return dt.DataTable(
        data=d["events"],
        columns=[{"name": c.replace("_"," ").title(), "id": c} for c in cols],
        style_table={"overflowX": "auto"},
        style_cell={"background": C["panel"], "color": C["text"],
                    "border": f"1px solid {C['border']}", "fontSize": "11px",
                    "padding": "7px 10px", "fontFamily": "Courier New"},
        style_header={"background": C["card"], "color": C["purple"],
                      "fontWeight": "bold", "border": f"1px solid {C['border']}",
                      "letterSpacing": "0.08em"},
        style_data_conditional=[
            {"if": {"filter_query": '{severity} > 0.7'}, "color": C["coral"]},
            {"if": {"filter_query": '{event_type} = "canal_closure"'}, "color": C["amber"]},
        ],
        page_size=8,
    )


@app.callback(Output("route-table","children"), Input("store","data"))
def cb_routes(d):
    if not d or not d["routes_df"]:
        return html.Div("Run QAOA to see route assignments.",
                        style={"color": C["muted"], "fontSize": "12px"})
    best_bs = d["best_bs"]
    meta    = d["meta"]
    rows    = d["routes_df"]
    # Mark selected routes
    for row in rows:
        var = row.get("var","")
        if var in meta["var_labels"]:
            qi = meta["var_labels"].index(var)
            row["selected"] = "✅ SELECTED" if qi < len(best_bs) and best_bs[qi]=="1" else ""
        else:
            row["selected"] = ""

    cols_show = ["vessel","route","distance_nm","days","cost_usd","var","selected"]
    return dt.DataTable(
        data=rows,
        columns=[{"name": c.replace("_"," ").title(), "id": c} for c in cols_show],
        style_table={"overflowX": "auto"},
        style_cell={"background": C["panel"], "color": C["text"],
                    "border": f"1px solid {C['border']}", "fontSize": "11px",
                    "padding": "7px 10px", "fontFamily": "Courier New"},
        style_header={"background": C["card"], "color": C["purple"],
                      "fontWeight": "bold", "border": f"1px solid {C['border']}"},
        style_data_conditional=[
            {"if": {"filter_query": '{selected} = "✅ SELECTED"'},
             "background": C["teal"]+"22", "color": C["cyan"]},
        ],
        page_size=12,
    )


if __name__ == "__main__":
    print("\n🚢  QAOA Maritime Logistics Dashboard")
    print("   Open → http://127.0.0.1:8051\n")
    app.run(debug=False, port=8051)

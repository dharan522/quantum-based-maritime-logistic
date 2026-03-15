import os, sys, json, threading, time
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd

from src.qubo_formulator  import MaritimeQUBOFormulator
from src.qaoa_solver      import MaritimeQAOASolver
from src.classical_solver import ClassicalMaritimeSolver
from src.digital_twin     import PortDigitalTwin

app   = Flask(__name__)
state = {"running":False,"logs":[],"progress":0,"done":False,"error":None,"report":None}

def log(msg, level="info"):
    state["logs"].append({"msg":msg,"level":level,"ts":time.strftime("%H:%M:%S")})

def run_thread(scenario, layers, vessels):
    try:
        state.update({"running":True,"logs":[],"progress":0,"done":False,"error":None,"report":None})
        import matplotlib; matplotlib.use("Agg")
        import numpy as np

        log("Loading data...","info")
        dist       = pd.read_csv("data/distance_matrix.csv", index_col=0)
        ports      = pd.read_csv("data/ports.csv",           index_col=0)
        vessels_df = pd.read_csv("data/vessels.csv")
        cargo      = pd.read_csv("data/cargo.csv")
        port_ids   = list(dist.index)
        state["progress"] = 10

        log(f"[1/6] Digital Twin — scenario={scenario}","step")
        twin     = PortDigitalTwin(port_ids, seed=2026)
        events   = twin.generate_disruptions(scenario=scenario, n_days=7)
        cong     = twin.simulate_congestion_heatmap(n_hours=48)
        berth_q  = twin.simulate_berth_queue("SHA", n_berths=8, n_hours=48)
        fuel_px  = twin.simulate_fuel_prices(n_days=30)
        arrivals = twin.simulate_arrivals(n_days=7)
        ev_df    = twin.get_event_summary()
        tp       = twin.compute_throughput_matrix(events)
        log(f"   {len(events)} events  |  avg throughput {tp['throughput_factor'].mean()*100:.1f}%","detail")
        state["progress"] = 22

        log(f"[2/6] QUBO formulation — vessels={vessels}","step")
        fml        = MaritimeQUBOFormulator(dist, vessels_df, cargo, ports, n_vessels=vessels, n_routes=3)
        fml.routes = twin.compute_route_cost_multipliers(fml.routes, events)
        Q, meta    = fml.build_qubo()
        routes_df  = fml.get_route_summary()
        log(f"   {meta['n_vars']} variables  |  shape {meta['qubo_shape']}","detail")
        state["progress"] = 35

        log(f"[3/6] QAOA optimisation — p={layers}, shots=2048","step")
        solver    = MaritimeQAOASolver(Q, meta["var_labels"], p_layers=layers, shots=2048)
        result    = solver.optimise(maxiter=150)
        qaoa_eval = fml.evaluate_solution(result["best_bitstring"])
        top_k     = solver.top_k_solutions(k=8)
        log(f"   Best: {result['best_bitstring']}","detail")
        log(f"   Energy: {qaoa_eval['qubo_energy']:.4f}  |  Cost: ${qaoa_eval['total_cost']:,.0f}","detail")
        log(f"   Converged: {'Yes' if result['converged'] else 'No'}","success" if result["converged"] else "warn")
        state["progress"] = 62

        log("[4/6] Classical benchmarks","step")
        cl     = ClassicalMaritimeSolver(dist, vessels_df, cargo, ports)
        greedy = cl.greedy_route_assignment(vessels)
        cw     = cl.clarke_wright_savings(n_vessels=vessels)
        bench  = cl.benchmark_summary(qaoa_eval["total_cost"], vessels)
        log(f"   Greedy: ${greedy['total_cost']:,.0f}  |  CW: ${cw['total_cost']:,.0f}","detail")
        state["progress"] = 72

        log("[5/6] Energy landscape scan","step")
        sub = min(6, Q.shape[0])
        sl1 = MaritimeQAOASolver(Q[:sub,:sub], meta["var_labels"][:sub], p_layers=1, shots=128)
        g, b, E = sl1.scan_landscape(resolution=14)
        state["progress"] = 82

        log("[6/6] Generating charts","step")
        from utils.charts import generate_all_charts
        generate_all_charts(solver, result, qaoa_eval, top_k, greedy, cw, bench,
                            cong, berth_q, arrivals, fuel_px, Q, meta, g, b, E,
                            scenario, port_ids, twin, events, fml, tp)
        log("   6 charts saved to outputs/","success")
        state["progress"] = 95

        report = {
            "scenario":scenario,"p_layers":layers,"n_vessels":vessels,
            "n_events":len(events),"qaoa":qaoa_eval,"top_k":top_k,
            "benchmark":bench.to_dict("records"),"qubo_meta":meta,
            "routes":routes_df.to_dict("records"),
            "events":ev_df.to_dict("records") if not ev_df.empty else [],
            "fuel_last":round(float(fuel_px[-1]),2),
        }
        with open("outputs/report.json","w") as f:
            json.dump(report, f, indent=2, default=str)

        state.update({"report":report,"progress":100,"done":True,"running":False})
        savings = greedy["total_cost"] - qaoa_eval["total_cost"]
        log("="*48,"info")
        log("PIPELINE COMPLETE","success")
        log(f"  QAOA cost   : ${qaoa_eval['total_cost']:>12,.0f}","result")
        log(f"  Greedy cost : ${greedy['total_cost']:>12,.0f}","result")
        log(f"  Difference  : ${savings:>+12,.0f}","result")
        log("="*48,"info")

    except Exception as e:
        import traceback
        state.update({"running":False,"error":str(e),"done":True})
        log(f"ERROR: {e}","error")
        log(traceback.format_exc(),"error")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_route():
    if state["running"]:
        return jsonify({"status":"already_running"}), 409
    d = request.json
    os.makedirs("outputs", exist_ok=True)
    threading.Thread(
        target=run_thread,
        args=(d.get("scenario","stressed"), int(d.get("layers",2)), int(d.get("vessels",4))),
        daemon=True
    ).start()
    return jsonify({"status":"started"})

@app.route("/status")
def status():
    return jsonify(state)

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "outputs"), filename)

if __name__ == "__main__":
    print("\n========================================")
    print("  QAOA Maritime Logistics — Web App")
    print("========================================")
    print("  Open → http://127.0.0.1:5000")
    print("========================================\n")
    app.run(debug=False, port=5000, threaded=True)
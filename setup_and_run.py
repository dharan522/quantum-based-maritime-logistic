#!/usr/bin/env python3
"""
setup_and_run.py
One-click setup: install → generate data → run tests → run pipeline → show results.

Usage:
    python setup_and_run.py
"""
import subprocess, sys, os

def run(cmd, desc):
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"{'─'*60}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"\n  ❌ Step failed: {cmd}")
        sys.exit(1)
    print(f"  ✅ Done")

print("\n" + "="*60)
print("  ⚓  QAOA Maritime Logistics — Auto Setup")
print("="*60)

run(f"{sys.executable} -m pip install -r requirements.txt -q",
    "1/4  Installing dependencies")

run(f"{sys.executable} data/generate_data.py",
    "2/4  Generating synthetic maritime dataset")

run(f"{sys.executable} -m pytest tests/ -v --tb=short",
    "3/4  Running 40 unit tests")

run(f"{sys.executable} main.py --scenario stressed --layers 2 --vessels 4",
    "4/4  Running full QAOA pipeline (scenario=stressed)")

print("\n" + "="*60)
print("  ✅  Setup complete!")
print()
print("  📊  Output charts:  outputs/")
print("  📋  Report:         outputs/report.json")
print()
print("  To launch the interactive dashboard:")
print("      python dashboard.py")
print("      → Open  http://127.0.0.1:8051")
print("="*60 + "\n")

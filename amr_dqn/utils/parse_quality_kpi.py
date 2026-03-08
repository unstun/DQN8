"""Helper: parse composite scores from table2_kpis_mean_filtered.csv for checkpoint screening."""
import csv
import sys

if len(sys.argv) < 2:
    print("?,N/A,N/A,N/A,N/A")
    sys.exit(0)

filepath = sys.argv[1]
kpi = {}
n_allsuc = "?"

try:
    with open(filepath) as f:
        for row in csv.DictReader(f):
            algo = row.get("Algorithm name", row.get("algo", "")).strip().lower()
            score = row.get("Composite score", row.get("composite_score", "")).strip()
            n = row.get("Filtered runs", "").strip()
            kpi[algo] = score
            if n and n_allsuc == "?":
                n_allsuc = n
except Exception as e:
    print(f"?,ERROR,ERROR,ERROR,ERROR  # {e}", file=sys.stderr)
    print("?,N/A,N/A,N/A,N/A")
    sys.exit(0)


def get(key):
    for k, v in kpi.items():
        if key in k:
            return v
    return "N/A"


print(f"{n_allsuc},{get('cnn-pddqn')},{get('cnn-ddqn')},{get('cnn-dqn')},{get('mlp-pddqn')}")
